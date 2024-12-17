import tkinter as tk

# 创建主窗口
root = tk.Tk()
root.title('Tkinter Example')

# 创建一个标签
label = tk.Label(root, text="Hello, Tkinter!")
label.pack()

# 定义按钮被点击时的函数
def callback():
    label.configure(text="Button was clicked!!")

# 创建一个按钮
button = tk.Button(root, text="Click Me", command=callback)
button.pack()

# 进入主循环
root.mainloop()