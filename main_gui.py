import os
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from model import *
root = Tk()
root.title("Neural Style Transfer")
root.geometry("800x450")
currdir=os.getcwd()
filename_s=""
filename_c=""
input_img_type="noise"

style_weight_ip=StringVar(root,"100000")
content_weight_ip=StringVar(root,"1")

optStep_ip=StringVar(root,"300")


cb_c1_var=StringVar(root,"")
cb_c2_var=StringVar(root,"")
cb_c3_var=StringVar(root,"")
cb_c4_var=StringVar(root,"")
cb_c5_var=StringVar(root,"")

cb_s1_var=StringVar(root,"")
cb_s2_var=StringVar(root,"")
cb_s3_var=StringVar(root,"")
cb_s4_var=StringVar(root,"")
cb_s5_var=StringVar(root,"")

def browsestyle():
    global filename_s
    filename_s = filedialog.askopenfilename(initialdir=currdir,
                                          title="Select a File",
                                          )
    label_file_explorer_S.configure(text="Style File Opened: " + filename_s)
    img = Image.open(filename_s)
    img = img.resize((50, 50), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.grid(row=0, column=3)
def browsecontent():
    global filename_c
    filename_c = filedialog.askopenfilename(initialdir=currdir,
                                          title="Select a File")
    label_file_explorer_c.configure(text="Content File Opened: " + filename_c)
    img = Image.open(filename_c)
    img = img.resize((50, 50), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.grid(row=1,column=3)

def input_image_rb(s):
    global input_img_type
    input_img_type = s
    print(s)
# def style_weight_ip()

def generate():
    content_layers,style_layers=get_content_style_layer_ip()
    style_img = image_loader(filename_s)
    content_img = image_loader(filename_c)
    style_weight= int(style_weight_ip.get())
    content_weight = int(content_weight_ip.get())
    opt_step=int(optStep_ip.get())
    print("style weight",style_weight)
    if input_img_type=="noise":
      input_img = torch.randn(content_img.data.size(), device=device)
    else:
      input_img = content_img.clone()

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img,
                                style_weight=style_weight,content_weight=content_weight,
                                num_steps=opt_step,
                                content_layer=content_layers,style_layer=style_layers
                                )
    # plt.figure()
    imshow(output, title='Output Image')
    img = Image.open(os.path.join(os.curdir,"styled_image_output.jpg"))
    img = img.resize((150, 150), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel = Label(root, image=img)
    panel.image = img
    panel.grid(row=5, column=3,columnspan=3,rowspan=3)

def get_content_style_layer_ip():
    layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    content_cb_status=[cb_c1_var.get(),cb_c2_var.get(),cb_c3_var.get(),cb_c4_var.get(),cb_c5_var.get()]
    style_cb_status = [cb_s1_var.get(), cb_s2_var.get(), cb_s3_var.get(), cb_s4_var.get(), cb_s5_var.get()]
    content_layers=[]
    style_layers=[]
    for (status,layer) in zip(content_cb_status,layers):
        if status=='1':
            content_layers.append(layer)
    for (status,layer) in zip(style_cb_status,layers):
        if status=='1':
            style_layers.append(layer)
    # print(content_layers,style_layers)
    return content_layers, style_layers



label_file_explorer_S = Label(root,
                            text = "Select Style file",
                            width = 50, height = 4,
                            fg = "blue")
label_file_explorer_c = Label(root,
                            text = "Select content file",
                            width = 50, height = 4,
                            fg = "blue")

button_explore_s = Button(root,
                        text="Browse Files",
                        command=browsestyle)
button_explore_c = Button(root,
                        text="Browse Files",
                        command=browsecontent)
style_w_label=Label(root,
                            text = "Style Weight",
                            width = 20, height = 4,
                            fg = "blue")
style_w = Entry(root,
                width = 10, textvariable=style_weight_ip
                )
content_w_label=Label(root,
                            text = "Content Weight",
                            width = 20, height = 4,
                            fg = "blue")
content_w = Entry(root,
                width = 10, textvariable=content_weight_ip
                )
optStep_label=Label(root,
                            text = "Optimization Steps",
                            width = 20, height = 4,
                            fg = "blue")
optStep = Entry(root,
                width = 10, textvariable=optStep_ip
                )
# style_w.insert(0,"100000")
button_generate = Button(root,
                        text="Generate Image",
                        command=generate)


v = StringVar(root, "noise")
init_label=Label(root,text="Initilize Input Image as")
radio_c=Radiobutton(root, text = "Content", variable = v,value="Content",command=lambda : input_image_rb("content"))
radio_n=Radiobutton(root, text = "Noise", variable = v,value="noise",command=lambda : input_image_rb("noise"))


label_cb_c=Label(root,text = "content_layer",width = 20, height = 4,fg = "blue")
cb_c1=Checkbutton(root,text="conv_1",variable=cb_c1_var)

cb_c2=Checkbutton(root,text="conv_2",variable=cb_c2_var)
cb_c3=Checkbutton(root,text="conv_3",variable=cb_c3_var)
cb_c4=Checkbutton(root,text="conv_4",variable=cb_c4_var)

cb_c5=Checkbutton(root,text="conv_5",variable=cb_c5_var)
cb_c1.deselect()
cb_c2.deselect()
cb_c3.deselect()
cb_c5.deselect()
cb_c4.select()



label_cb_s=Label(root,text = "style_layer",width = 20, height = 4,fg = "blue")
cb_s1=Checkbutton(root,text="conv_1",variable=cb_s1_var)

cb_s2=Checkbutton(root,text="conv_2",variable=cb_s2_var)
cb_s3=Checkbutton(root,text="conv_3",variable=cb_s3_var)
cb_s4=Checkbutton(root,text="conv_4",variable=cb_s4_var)

cb_s5=Checkbutton(root,text="conv_5",variable=cb_s5_var)
cb_s1.select()
cb_s2.select()
cb_s3.select()
cb_s5.select()
cb_s4.select()

label_about=Label(root,text = "NST algorithm by Leon A. Gatys implementd using VGG19",width = 80, height = 4,fg = "black")

# print(v)
button_exit = Button(root,
                     text="Exit",
                     command=exit)


label_file_explorer_S.grid(column = 0, row = 0)
label_file_explorer_c.grid(column = 0, row = 1)
button_explore_s.grid(column=1, row=0)
button_explore_c.grid(column=1, row=1)


label_cb_c.grid(row=2,column=0)
cb_c1.grid(row=2,column=1)
cb_c2.grid(row=2,column=2)
cb_c3.grid(row=2,column=3)
cb_c4.grid(row=2,column=4)
cb_c5.grid(row=2,column=5)


label_cb_s.grid(row=3,column=0)
cb_s1.grid(row=3,column=1)
cb_s2.grid(row=3,column=2)
cb_s3.grid(row=3,column=3)
cb_s4.grid(row=3,column=4)
cb_s5.grid(row=3,column=5)


init_label.grid(row=4,column=0)
radio_c.grid(row=4,column=1)
radio_n.grid(row=4,column=2)

style_w_label.grid(row=5,column=0)
style_w.grid(row=5,column=1)
content_w_label.grid(row=6,column=0)
content_w.grid(row=6,column=1)

optStep_label.grid(row=7,column=0)
optStep.grid(row=7,column=1)

button_generate.grid(row=8,column=0)

button_exit.grid(column=1, row=8)

label_about.grid(column=0,row=9,columnspan=4)





root.mainloop()