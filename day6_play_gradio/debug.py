import gradio as gr

def test_fn(img):
    print("收到图片：", type(img))
    return img

iface = gr.Interface(fn=test_fn, inputs=gr.Image(type="pil"), outputs="image")
iface.launch(share=True, debug=True)