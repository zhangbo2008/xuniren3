import gradio as gr

# 定义处理函数/模型推理
def greet(name):
    return "Hello " + name + "!"

# interface中三个必要参数: 推理函数、输入、输出
demo = gr.Interface(fn=greet, inputs="text", outputs="text")
# 是否开启公网分享
demo.launch(server_name='0.0.0.0',share=True)