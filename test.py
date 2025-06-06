from openai import OpenAI
import json
from ansys.aedt.core import Maxwell3d
import numpy as np
import xlwings as xw

project_path = r"Project23.aedt"  # 设计原型路径
coil_calculator_path = '线圈损耗计算器.xlsx'  # 线圈损耗计算器路径，线圈损耗使用场计算器获取H2积分后通过调用excl表格计算器获得损耗结果

"""以下参数为系统实际工作补偿参数，用于根据互感解算修正激励，获得准确损耗值"""
w = 2 * np.pi * 85000
R = 10 * 8 / (np.pi ** 2)
L2 = 7.1 * 10 ** (-6)
C2 = 493.8 * 10 ** (-9)
P = 3300
I2 = w * C2 * np.sqrt(P * R)

"""磁芯块尺寸，100mm * 100mm * 5mm"""
unit_size_x = 100
unit_size_y = 100
thickness = 5  # mm

"""磁芯块起始xy坐标"""
start_x = -350
start_y = -250

"""存储PB曲线"""
points_at_frequency: dict = {}

"""LLM返回数据的key"""
required_keys = ["Toplayer", "Middlelayer", "Bottomlayer"]


def has_required_keys(params, required_keys):
    """检查params是否包含所有必需的key"""
    return all(key in params for key in required_keys)


def parse_tab_to_frequency_dict(file_path, freq_hz):
    """解析标准 BP 曲线 .Tab 文件为字典格式

    :param file_path: .Tab 文件路径
    :param freq_hz: 文件表示的频率点
    :return: 格式如 {100: [[x1,y1], [x2,y2], ...]} 的字典
    """
    try:
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]  # 过滤空行

            # 解析频率标题行（支持两种格式）

            data_points = []
            for line in lines[0:]:
                parts = line.split()
                if len(parts) < 2:
                    continue

                # 数值类型智能转换（整数保留精度）
                x = float(parts[0])
                y = float(parts[1])

                data_points.append([x, y])

            points_at_frequency[freq_hz] = data_points

    except Exception as e:
        print(f"文件解析错误: {str(e)}")
        return {}


parse_tab_to_frequency_dict("DMR95-100kHZ.tab", 100000)
parse_tab_to_frequency_dict("DMR95-200kHZ.tab", 200000)
parse_tab_to_frequency_dict("DMR95-300kHZ.tab", 300000)
parse_tab_to_frequency_dict("DMR95-500kHZ.tab", 500000)
parse_tab_to_frequency_dict("DMR95-1MHZ.tab", 1000000)
print("解析结果:")
print(points_at_frequency)

"""检验LLM返回设计的磁芯用量"""


def check_design(matrix1, matrix2, matrix3):
    num = 0
    for (x, y) in np.argwhere(matrix1 == 1):
        num = num + 1
    for (x, y) in np.argwhere(matrix2 == 1):
        num = num + 1
    for (x, y) in np.argwhere(matrix3 == 1):
        num = num + 1
    return num


class DeepSeekOptimizer:
    def __init__(self,
                 ds_api_key: str,
                 optimization_target: str = 'minimize',
                 max_iterations: int = 50,
                 exploration_rate: float = 0.2):
        """
        参数优化控制器

        :param ds_api_key: DeepSeek API密钥
        :param optimization_target: 优化目标 'maximize' 或 'minimize'
        :param max_iterations: 最大迭代次数
        :param exploration_rate: 探索率（0-1，越高越倾向随机探索）
        """
        # 初始化API客户端，本样例使用硅基流动
        self.client = OpenAI(api_key=ds_api_key, base_url='https://api.siliconflow.cn/v1/')

        # 配置参数
        self.target = optimization_target
        self.max_iter = max_iterations
        self.explore_rate = exploration_rate
        self.history = []
        self.best = {
            'score': -np.inf if self.target == 'maximize' else np.inf,
            'params': {}
        }

    def _construct_prompt(self) -> list:
        """构建优化提示"""
        last_results = self.history

        system_msg = {
            "role": "system", "content": f"""您是一个专业的磁芯结构设计优化引擎，现在要设计无线充电发射端的磁芯，
            所设计结构所处环境上方为发射和接收线圈，下方为发射端的屏蔽铝板，设计结构参数以三个7*5的布尔矩阵形式表示，
            共三层，请根据仿真历史参数和损失分数（分数越小越好），设计并探索更优矩阵。要求： 
            1. 必须生成严格合法的JSON对象 
            2. 参数必须为三个7*5的布尔矩阵(仅包含0和1)，每个矩阵内部尽量中心对称 
            3. 输出前请严格验算矩阵中1的数目为35，且不能与任意历史设计重复，不满足该条件的设计矩阵无效
            4. 三个矩阵中的1的可以均匀分配也可以失衡分配（矩阵可以为全零），适当平衡探索与利用 
            5. 根据历史记录，每次仅推荐一个方案，且不能与历史记录重复，严格以""Toplayer":矩阵"、""Middlelayer":矩阵"、""Bottomlayer":矩阵"和""大体迭代思路":"XXX""形式返回
            6. 使用先进的参数优化算法，每次推荐参数给出中文迭代思路，包含矩阵参数迭代计算过程"""}

        user_msg = {
            "role": "user",
            "content": f"""历史记录：
                {last_results}

            请给出推荐参数方案（JSON格式）："""}

        return [system_msg, user_msg]

    def _parse_recommendation(self, response) -> dict:
        """解析AI推荐参数"""
        try:
            # 提取JSON内容
            json_str = response.choices[0].message.content.split("```json")[1].split("```")[0]
            params = json.loads(json_str)
            return params
        except Exception as e:
            print(f"参数解析失败: {str(e)}，启用随机回退")
            return self._random_params()

    def _random_params(self) -> dict:
        """生成随机参数"""
        return {
            "param": np.random.randint(0, 2, (7, 5))
        }

    def _call_simulation(self, param, matrix1, matrix2, matrix3, number):
        """调用仿真"""
        try:
            m3d = Maxwell3d(
                project=project_path,
                close_on_exit=False
            )
            """仿真失败弃用设计，score置1000"""
            scores = 1000
            try:
                # 定义铁氧体材料
                if not m3d.materials.exists_material("Ferrite_DMR95"):
                    mat = m3d.materials.add_material("Ferrite_DMR95")

                    mat.set_coreloss_at_frequency(points_at_frequency=points_at_frequency, conductivity=0.1,
                                                  coefficient_setup="w_per_cubic_meter",
                                                  core_loss_model_type="Power Ferrite")
                    mat.permeability = 3300
                    mat.conductivity = 0.1
                    print(len(points_at_frequency))

                # 生成磁芯几何-------------------------------
                core_objects = []
                core_name = []
                for (x, y) in np.argwhere(matrix1 == 1):
                    box1 = m3d.modeler.create_box(
                        origin=[start_x + x * unit_size_x, start_y + y * unit_size_y, 38],
                        sizes=[unit_size_x, unit_size_y, -thickness],
                        name=f"CoreBlock_top_{x}_{y}",
                        material="Ferrite_DMR95"
                    )
                    core_name.append(f"CoreBlock_top_{x}_{y}")
                    core_objects.append(box1)
                for (x, y) in np.argwhere(matrix2 == 1):
                    box2 = m3d.modeler.create_box(
                        origin=[start_x + x * unit_size_x, start_y + y * unit_size_y, 38 - thickness - 1],
                        sizes=[unit_size_x, unit_size_y, -thickness],
                        name=f"CoreBlock_middle_{x}_{y}",
                        material="Ferrite_DMR95"
                    )
                    core_name.append(f"CoreBlock_middle_{x}_{y}")
                    core_objects.append(box2)
                for (x, y) in np.argwhere(matrix3 == 1):
                    box3 = m3d.modeler.create_box(
                        origin=[start_x + x * unit_size_x, start_y + y * unit_size_y, 38 - thickness - thickness - 2],
                        sizes=[unit_size_x, unit_size_y, -thickness],
                        name=f"CoreBlock_bottom_{x}_{y}",
                        material="Ferrite_DMR95"
                    )
                    core_name.append(f"CoreBlock_bottom_{x}_{y}")
                    core_objects.append(box3)
                # 合并所有磁芯单元
                # if core_objects:
                #    m3d.modeler.unite(core_objects, purge=True)

                # 创建空气区域包围盒
                '''
                air_margin = 100  # mm
                m3d.modeler.create_air_region(
                    x_pos=air_margin,
                    y_pos=air_margin,
                    z_pos=air_margin,
                    x_neg=air_margin,
                    y_neg=air_margin,
                    z_neg=air_margin
                )
                '''
                m3d.set_core_losses(assignment=core_name)
                m3d.set_core_losses(assignment=["S_core"])

                winding_P = m3d.assign_winding(winding_type="Current", current=20, phase=0, name="Winding_P",
                                               is_solid=False)
                m3d.assign_winding(winding_type="Current", current=I2, phase=90, name="Winding_S",
                                   is_solid=False)

                m3d.add_winding_coils(assignment="Winding_P", coils=["CoilTerminal1"])
                m3d.add_winding_coils(assignment="Winding_S", coils=["CoilTerminal2"])

                m3d.assign_matrix(assignment=["Winding_P", "Winding_S"], matrix_name="Matrix_1")

                mesh = m3d.mesh
                for element in core_name:
                    mesh.assign_length_mesh(
                        assignment=[element],
                        maximum_length=10,  # 基础单元尺寸 (根据磁芯尺寸调整)
                        maximum_elements=1000
                    )

                mesh.assign_length_mesh(
                    assignment=["S_core"],
                    maximum_length=10,  # 基础单元尺寸 (根据磁芯尺寸调整)
                    maximum_elements=10000
                )
                mesh.generate_mesh(name="Setup1")  # 生成网格

                m3d.analyze(setup="Setup1", cores=4, tasks=4)  #预FEA，获取耦合条件，互感
                solution_data = m3d.post.get_solution_data(
                    expressions=["CplCoef(Winding_S,Winding_P)", "L(Winding_P,Winding_S)"],
                    context="Matrix_1")
                print(solution_data.data_magnitude(expression="CplCoef(Winding_S,Winding_P)"))
                print(solution_data.data_magnitude(expression="L(Winding_P,Winding_S)"))
                M = solution_data.data_magnitude(expression="L(Winding_P,Winding_S)")[0] * 10 ** (-9)
                # 修正激励
                I1 = np.sqrt(P * L2 / (w ** 2 * M ** 2 * R * C2))
                winding_P.update_property(prop_name="current", prop_value=f"{I1}A")

                m3d.analyze(setup="Setup1", cores=4, tasks=4)
                solution_data1 = m3d.post.get_solution_data(expressions=["CoreLoss", "SolidLoss"],
                                                            context="Matrix_1")
                solution_data2 = m3d.post.get_solution_data(report_category="Fields",
                                                            expressions=["P_coil_H2", "S_coil_H2"],
                                                            variations={"Phase": "0deg"})

                CoreLoss = solution_data1.data_magnitude(expression="CoreLoss")[0] / 1000
                SolidLoss = solution_data1.data_magnitude(expression="SolidLoss")[0] / 1000
                P_Coil_H2 = solution_data2.data_magnitude(expression="P_coil_H2")[0]
                S_Coil_H2 = solution_data2.data_magnitude(expression="S_coil_H2")[0]

                print(CoreLoss)
                print(SolidLoss)
                print(P_Coil_H2)
                print(S_Coil_H2)

                app = xw.App(visible=False)  # 不显示Excel界面
                wb = app.books.open('coilloss_calculator.xlsx')
                wb.sheets[0].range('H5').value = I1
                wb.sheets[0].range('H6').value = I2
                wb.sheets[0].range('J5').value = P_Coil_H2
                wb.sheets[0].range('J6').value = S_Coil_H2
                P_CoilLoss = wb.sheets[0].range('T5').value
                S_CoilLoss = wb.sheets[0].range('T6').value

                print(
                    f"发射端线圈损耗 {P_CoilLoss} W，接收端线圈损耗 {S_CoilLoss} W, 磁芯损耗 {CoreLoss} W，铝板损耗 {SolidLoss} W，"
                    f"总损耗 {P_CoilLoss + S_CoilLoss + CoreLoss + SolidLoss}")

                # 关闭工作簿和应用
                wb.save()
                wb.close()
                app.quit()
                scores = P_CoilLoss + S_CoilLoss + CoreLoss + SolidLoss

            except Exception as e:
                print(f"仿真过程中出现错误: {str(e)}")
            finally:
                if self._update_state(param, scores, number):
                    m3d.save_project(
                        file_name=f"D:\LLM_design\simulation_file\history_file_1\iter_{number}\iter_{number}.aedt")
                m3d.release_desktop()

            return scores
        except Exception as e:
            print(f"仿真调用失败: {str(e)}")
            return None

    def _update_state(self, params: dict, score: float, number):
        """更新优化状态"""
        his = {
            "params": {"Toplayer": params["Toplayer"],
                       "Middlelayer": params["Middlelayer"],
                       "Bottomlayer": params["Bottomlayer"]
                       },
            "score": score,
            "iteration": len(self.history) + 1
        }
        with open(f"D:\LLM_design\simulation_file\history_file_1\iter_output{number}.txt", 'w') as f:
            for key, value in his.items():
                f.write(f"{key}: {value}\n")  # 格式为 key: value
        self.history.append(his)

        # 更新最佳记录
        if (self.target == 'maximize' and score > self.best['score']) or \
                (self.target == 'minimize' and score < self.best['score']):
            self.best.update(params=params, score=score)
            return True
        else:
            return False

    def optimize(self) -> dict:
        """执行优化主循环"""

        for num in range(self.max_iter):
            # 1. 生成推荐参数
            prompt = self._construct_prompt()
            print(prompt)
            while True:
                # 1. 发送请求获取推荐参数，使用deepseek R1模型
                response = self.client.chat.completions.create(
                    model="Pro/deepseek-ai/DeepSeek-R1",
                    messages=prompt,
                    max_tokens=16384,
                    stream=False
                )

                # 2. 解析推荐参数
                print(response.choices[0].message.content)
                history_matrices = []  # 用于存储历史的三矩阵组合

                try:
                    param = response.choices[0].message.content.split("```json")[1].split("```")[0]
                    params = json.loads(param)
                    print(params)
                    if not has_required_keys(params, required_keys):
                        print(f"验证失败，param缺少必需的key（需要{required_keys}），将重新请求...")
                        continue

                    matrix1 = np.array(params["Toplayer"])
                    print(matrix1)
                    matrix2 = np.array(params["Middlelayer"])
                    print(matrix2)
                    matrix3 = np.array(params["Bottomlayer"])
                    print(matrix3)

                    # 检查是否与历史矩阵重复
                    is_duplicate = False
                    for hist_matrices in history_matrices:
                        if (np.array_equal(matrix1, hist_matrices[0]) and
                                np.array_equal(matrix2, hist_matrices[1]) and
                                np.array_equal(matrix3, hist_matrices[2])):
                            is_duplicate = True
                            break
                    if is_duplicate:
                        print("发现重复的设计组合，将重新请求...")
                        continue

                    # 边界条件：检查磁芯的总数是否为35
                    total_ones = check_design(matrix1, matrix2, matrix3)
                    if total_ones == 35:
                        print(f"验证通过，三个矩阵中共有{total_ones}个1")
                        history_matrices.append((matrix1.copy(), matrix2.copy(), matrix3.copy()))
                        score = self._call_simulation(param=params, matrix1=matrix1, matrix2=matrix2, matrix3=matrix3,
                                                      number=num)
                        print(f"Iter {len(self.history)}: Score={score:.4f} Params={params}")
                        break
                    else:
                        print(f"验证失败，三个矩阵中共有{total_ones}个1（需要35个），将重新请求...")

                except (IndexError, json.JSONDecodeError, KeyError) as e:
                    print(f"解析响应时出错: {e}，将重新请求...")
                    continue

        return self.best


# 替换ds_api_key为自己的访问密钥
optimizer = DeepSeekOptimizer(
    ds_api_key="*********************",
    optimization_target="minimize",
    max_iterations=100
)

result = optimizer.optimize()
print(f"\n最优参数: {result['params']}")
print(f"最佳得分: {result['score']:.2f}")
