import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
from PIL import Image, ImageTk
import main  # Your business logic module

class TextPopup(tk.Toplevel):
    def __init__(self, parent, title, message):
        super().__init__(parent)
        self.title(title)
        self.geometry("600x400")  # Adjust size as needed

        # Text area
        text = tk.Text(self, wrap='word')
        text.insert('1.0', message)
        text.configure(state='disabled')  # Read-only, allow selecting and copying
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text['yscrollcommand'] = scrollbar.set

        self.protocol("WM_DELETE_WINDOW", self.destroy)

class LoginWindow:
    """登录窗口类"""
    def __init__(self, root, on_success):
        """
        登录窗口初始化
        :param root: 主窗口（将被隐藏，直到登录成功）
        :param on_success: 登录成功后的回调函数
        """
        self.root = root
        self.login_window = tk.Toplevel(root)
        self.login_window.title("Login")
        self.login_window.geometry("400x300")
        self.login_window.resizable(False, False)

        # 居中窗口
        window_width, window_height = 400, 300
        screen_width = self.login_window.winfo_screenwidth()
        screen_height = self.login_window.winfo_screenheight()
        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))
        self.login_window.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # 登录界面标题
        tk.Label(self.login_window, text="Login", font=("Arial", 24, "bold")).pack(pady=20)

        # 用户名标签和输入框
        tk.Label(self.login_window, text="Username:", font=("Arial", 12)).pack(pady=5)
        self.username_entry = tk.Entry(self.login_window, font=("Arial", 12))
        self.username_entry.pack(pady=5)

        # 密码标签和输入框
        tk.Label(self.login_window, text="Password:", font=("Arial", 12)).pack(pady=5)
        self.password_entry = tk.Entry(self.login_window, show="*", font=("Arial", 12))
        self.password_entry.pack(pady=5)

        # 登录按钮
        self.login_button = tk.Button(self.login_window, text="Login", font=("Arial", 12), command=self.check_login)
        self.login_button.pack(pady=20)

        # 错误提示标签
        self.error_label = tk.Label(self.login_window, text="", font=("Arial", 10), fg="red")
        self.error_label.pack()

        self.on_success = on_success

    def check_login(self):
        """验证用户名和密码"""
        username = self.username_entry.get().strip()
        password = self.password_entry.get().strip()

        if username == "CHG" and password == "123456":
            messagebox.showinfo("Success", "Login successful!")
            self.login_window.destroy()  # 关闭登录窗口
            self.on_success()  # 触发成功回调函数
        else:
            self.error_label.config(text="Invalid username or password")

class App:
    def __init__(self, root):
        self.root = root
        root.title("CHG iTruck Optimizer")

        # Default paths
        self.default_paths = {
            'city_location': "./data source/city-location.csv",
            'seg_division': "./data source/segment-division.csv",
            'rental_truck_list': "./data source/Rental truck list.csv",
            'exchange_rate': "./data source/exchange rate.xlsx",
            'LTC_cost': "./data source/callout cost.csv",
            'equipment_vehicle_type': "./data source/equipment id-vehicle type.csv",
            'special_lane_cost': "./data source/special_lane cost.csv",
            'transit_time': "./data source/调车时间.csv",
            'rental_cost': "./data source/rental cost.csv",
            'source_dir': "./data source",
            'processed_data_path': "./results/processed_data.csv",
            'main_lane_forecast_path': "./data source/Main Lane Operation Forecast.csv",
            'Predict_output_dir': "predict results",
        }

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.file_path = None
        self.output_dir = None

        self.build_gui()

    def show_text_popup(self, title, message):
        popup = TextPopup(self.root, title, message)
        popup.grab_set()  # Modal window

    def build_gui(self):
        from PIL import Image, ImageTk

        # 最大化窗口，带标题栏和任务栏
        try:
            self.root.state('zoomed')  # Windows & 部分Linux
        except:
            try:
                self.root.attributes('-zoomed', True)  # macOS & 一些Linux
            except:
                self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()}")

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

        # 顶部区域 - 图片和标题
        top_height = self.screen_height // 3
        top_frame = tk.Frame(self.root, width=self.screen_width, height=top_height)
        top_frame.pack_propagate(False)
        top_frame.pack(fill=tk.X, side=tk.TOP)

        # 加载并调整图片
        image_path = "image.jpg"  # 换成你的图片路径
        img = Image.open(image_path)
        try:
            resample_method = Image.Resampling.LANCZOS
        except AttributeError:
            resample_method = Image.ANTIALIAS
        img = img.resize((self.screen_width, top_height), resample_method)
        self.header_img = ImageTk.PhotoImage(img)

        canvas = tk.Canvas(top_frame, width=self.screen_width, height=top_height, highlightthickness=0, bd=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvas.create_image(0, 0, anchor='nw', image=self.header_img)
        # 居中标题
        canvas.create_text(self.screen_width // 2, top_height // 2,
                           text="CHG iTruck Optimizer",
                           font=("Arial", 42, "bold"),
                           fill="white")

        # 中间区域 - 表单与按钮
        middle_frame = tk.Frame(self.root)
        middle_frame.pack(padx=10, pady=10, anchor="n")

        # 选择主输入文件
        tk.Label(middle_frame, text="Select Main Input File:").grid(row=0, column=0, sticky='w')
        self.entry_file = tk.Entry(middle_frame, width=50)
        self.entry_file.grid(row=0, column=1)
        tk.Button(middle_frame, text="Browse", command=self.select_file).grid(row=0, column=2, padx=5)

        # 选择输出目录
        tk.Label(middle_frame, text="Select Output Directory:").grid(row=1, column=0, sticky='w', pady=5)
        self.entry_output = tk.Entry(middle_frame, width=50)
        self.entry_output.grid(row=1, column=1)
        tk.Button(middle_frame, text="Browse", command=self.select_dir).grid(row=1, column=2, padx=5)

        # 修改默认路径按钮
        tk.Button(middle_frame, text="Modify Default Paths", command=self.open_modify_paths_window).grid(row=2,
                                                                                                         column=0,
                                                                                                         pady=10)

        # 开始、截止日期
        tk.Label(middle_frame, text="Start Date (YYYY-MM-DD):").grid(row=2, column=1, sticky='e')
        self.start_date_entry = tk.Entry(middle_frame, width=12)
        self.start_date_entry.grid(row=2, column=2, padx=5, sticky='w')
        self.start_date_entry.insert(0, "2025-01-01")
        tk.Button(middle_frame, text="Submit Start Date", command=self.submit_start_date).grid(row=2, column=3, padx=5)

        tk.Label(middle_frame, text="End Date (YYYY-MM-DD):").grid(row=2, column=4, sticky='e')
        self.end_date_entry = tk.Entry(middle_frame, width=12)
        self.end_date_entry.grid(row=2, column=5, padx=5, sticky='w')
        self.end_date_entry.insert(0, "2025-06-30")
        tk.Button(middle_frame, text="Submit End Date", command=self.submit_end_date).grid(row=2, column=6, padx=5)

        # 主功能按钮区
        btn_frame = tk.Frame(middle_frame)
        btn_frame.grid(row=3, column=0, columnspan=7, pady=10)

        tk.Button(btn_frame, text="Data Preprocessing & Quality Check", width=30,
                  command=self.run_data_preprocessing).pack(pady=3)
        tk.Button(btn_frame, text="Rental to LTC Fee Calculation", width=30, command=self.run_fee_compare).pack(
            pady=3)
        tk.Button(btn_frame, text="Rental Truck Time Utilization", width=30, command=self.run_rental_utilization).pack(
            pady=3)
        tk.Button(btn_frame, text="LTC Trip Concatenation", width=30, command=self.run_callout_concat).pack(pady=3)
        tk.Button(btn_frame, text="LTC to Rental Fee Simulation", width=30,
                  command=self.run_callout_to_rental).pack(pady=3)
        # Prediction按钮紧接着主功能按钮
        tk.Button(btn_frame, text="LTC & Rental Prediction", width=30, command=self.run_prediction).pack(pady=3)

        # 状态标签
        self.status_label = tk.Label(self.root, text="Waiting for operation…")
        self.status_label.pack(pady=10, anchor='s')

    def submit_start_date(self):
        start_date_str = self.start_date_entry.get().strip()
        if start_date_str == "":
            self.selected_start_date = None
            messagebox.showinfo("Info", "No start date entered. Processing will not filter by start date.")
        else:
            import datetime
            try:
                datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
                self.selected_start_date = start_date_str
                messagebox.showinfo("Info", f"Start date submitted: {start_date_str}")
            except ValueError:
                messagebox.showerror("Error", "Start date format is incorrect. Please enter as YYYY-MM-DD.")
                self.selected_start_date = None

    def submit_end_date(self):
        end_date_str = self.end_date_entry.get().strip()
        if end_date_str == "":
            self.selected_end_date = None
            messagebox.showinfo("Info", "No end date entered. Processing will not filter by end date.")
        else:
            import datetime
            try:
                datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
                self.selected_end_date = end_date_str
                messagebox.showinfo("Info", f"End date submitted: {end_date_str}")
            except ValueError:
                messagebox.showerror("Error", "End date format is incorrect. Please enter as YYYY-MM-DD.")
                self.selected_end_date = None

    def select_file(self):
        file = filedialog.askopenfilename(title="Select Main Input File", filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")])
        if file:
            self.file_path = file
            self.entry_file.delete(0, tk.END)
            self.entry_file.insert(0, file)

    def select_dir(self):
        dir = filedialog.askdirectory(title="Select Output Directory")
        if dir:
            self.output_dir = dir
            self.entry_output.delete(0, tk.END)
            self.entry_output.insert(0, dir)

    def open_modify_paths_window(self):
        """Popup window to modify default paths"""
        win = tk.Toplevel(self.root)
        win.title("Modify Default Paths")
        win.geometry("700x400")

        entries = {}

        def browse_path(key):
            f = filedialog.askopenfilename(
                title=f"Select {key} File",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
            )
            if f:
                entries[key].delete(0, tk.END)
                entries[key].insert(0, f)

        row = 0
        for key, path in self.default_paths.items():
            tk.Label(win, text=key, width=25, anchor='w').grid(row=row, column=0, padx=5, pady=3)
            entry = tk.Entry(win, width=60)
            entry.grid(row=row, column=1, padx=5, pady=3)
            entry.insert(0, path)
            entries[key] = entry
            btn = tk.Button(win, text="Browse", command=lambda k=key: browse_path(k))
            btn.grid(row=row, column=2, padx=5, pady=3)
            row += 1

        def save_and_close():
            for k, entry in entries.items():
                new_path = entry.get().strip()
                if new_path:
                    self.default_paths[k] = new_path
            messagebox.showinfo("Info", "Default paths updated!")
            win.destroy()

        btn_save = tk.Button(win, text="Save", width=15, command=save_and_close)
        btn_save.grid(row=row, column=1, pady=15)

    def check_paths(self):
        if not self.file_path or not os.path.isfile(self.file_path):
            messagebox.showwarning("Warning", "Please select the main input file!")
            return False
        if not self.output_dir or not os.path.isdir(self.output_dir):
            messagebox.showwarning("Warning", "Please select the output directory!")
            return False
        return True

    def run_in_thread(self, func):
        def wrapper():
            try:
                self.status_label.config(text="Running, please wait…")
                func()
                self.status_label.config(text="Operation completed!")
                messagebox.showinfo("Done", "Operation completed successfully!")
            except Exception as e:
                self.status_label.config(text="Error occurred during execution")
                messagebox.showerror("Error", str(e))
        threading.Thread(target=wrapper).start()

    def run_data_preprocessing(self):
        if self.selected_start_date is None:
            ret = messagebox.askyesno("Prompt", "Start date not submitted. Continue without start date filter?")
            if not ret:
                self.status_label.config(text="Waiting for operation…")
                return
        if self.selected_end_date is None:
            ret = messagebox.askyesno("Prompt", "End date not submitted. Continue without end date filter?")
            if not ret:
                self.status_label.config(text="Waiting for operation…")
                return

        if not self.check_paths():
            return

        start_date_str = self.selected_start_date
        end_date_str = self.selected_end_date

        def task():
            output_file, msg = main.perform_data_preprocessing(
                self.file_path,
                self.default_paths['city_location'],
                self.default_paths['seg_division'],
                self.default_paths['special_lane_cost'],
                self.default_paths['transit_time'],
                self.default_paths['source_dir'],
                self.output_dir,
                self.default_paths['rental_truck_list'],
                start_date_str=self.selected_start_date,
                end_date_str=self.selected_end_date
            )
            if msg:
                self.show_text_popup("Data Quality Check Result", msg)

            main.split_rental_callout(
                output_file,
                self.default_paths['rental_truck_list'],
                self.output_dir
            )

        self.run_in_thread(task)

    def run_fee_compare(self):
        if not self.check_paths():
            return

        def task():
            main.perform_fee_comparison(
                self.output_dir,
                self.default_paths['rental_truck_list'],
                self.default_paths['exchange_rate'],
                self.default_paths['LTC_cost'],
                self.default_paths['special_lane_cost'],
                self.default_paths['equipment_vehicle_type'],
                self.default_paths['rental_cost'],
            )
        self.run_in_thread(task)

    def run_rental_utilization(self):
        if not self.check_paths():
            return
        self.run_in_thread(lambda: main.perform_rental_time_utilization(
            self.output_dir,
            self.default_paths['rental_cost'],
            self.default_paths['LTC_cost'],
            self.default_paths['rental_truck_list'],
            self.default_paths['exchange_rate'],
            ))

    def run_callout_concat(self):
        if not self.check_paths():
            return
        self.run_in_thread(lambda: main.perform_callout_concat(self.output_dir, self.default_paths['transit_time']))

    def run_callout_to_rental(self):
        if not self.check_paths():
            return
        self.run_in_thread(lambda: main.perform_callout_to_rental(
            self.output_dir,
            self.default_paths['rental_cost'],
            self.default_paths['exchange_rate'],
            self.default_paths['transit_time']
        ))

    def run_prediction(self):
        if not self.check_paths():
            return

        def task():
            predict_dir = os.path.join(self.output_dir, "predict results")
            os.makedirs(predict_dir, exist_ok=True)  # 确保目录存在
            # 提前拼接好所有输出路径
            rental_to_callout_compare_path = os.path.join(predict_dir, 'future_rental_compare.csv')
            rental_time_util_output_path = os.path.join(predict_dir, 'rental_time_utilization_output.csv')
            output_path_callout_cost = os.path.join(predict_dir, 'callout_cost_output.csv')
            output_path_special_lane_cost = os.path.join(predict_dir, 'special_lane_cost_output.csv')
            # 调用 main.py 中的预测函数
            main.perform_rental_reduction(
                self.default_paths['processed_data_path'],
                self.default_paths['main_lane_forecast_path'],
                self.default_paths['rental_truck_list'],
                self.default_paths['exchange_rate'],
                self.default_paths['LTC_cost'],
                self.default_paths['special_lane_cost'],
                self.default_paths['equipment_vehicle_type'],
                output_path_callout_cost,
                self.default_paths['rental_cost'],
                output_path_special_lane_cost,
                rental_to_callout_compare_path,
                rental_time_util_output_path,
                predict_dir
            )
            main.perform_callout_predict(
                self.default_paths['processed_data_path'],
                self.default_paths['rental_truck_list'],
                self.default_paths['main_lane_forecast_path'],
                self.default_paths['LTC_cost'],
                self.default_paths['special_lane_cost'],
                self.default_paths['exchange_rate'],
                self.default_paths['transit_time'],
                self.default_paths['rental_cost'],
                predict_dir
            )

        self.run_in_thread(task)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口，直到登录成功

    def start_main_app():
        """启动主应用程序"""
        root.deiconify()  # 显示主窗口
        app = App(root)

    # 显示登录窗口
    LoginWindow(root, start_main_app)
    root.mainloop()