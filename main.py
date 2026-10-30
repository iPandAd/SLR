import cv2
import datetime
from keyboard import is_pressed
import os
import PySimpleGUI as sg
import pyttsx3
import sys
from torch import load
from torch import Tensor
from torch import topk
from torch.optim import Adam

import config
from model import ResNet18


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def main():
    sg.theme("BrownBlue")

    parse = "Launch application".center(80, "=")
    parse += "\nPress key to:\n"
    parse += "'Esc': quit\t|\t's': save images\n"
    parse += "'Enter': add letter\t|\t'Backspace': delete letter\n"
    parse += "' ': split words\t|\t'Tab': speak out\n"
    parse += "'1' - '5': quick response\n"

    # 定义窗口布局
    layout = [
        [sg.Button("Save", size=(10, 1), font="Courier"),
         sg.Button("Add", size=(10, 1), font="Courier"),
         sg.Button("Delete", size=(10, 1), font="Courier"),
         sg.Button("Split", size=(10, 1), font="Courier"),
         sg.Button("Speak", size=(10, 1), font="Courier"),
         sg.Button("Exit", size=(10, 1), font="Courier")],
        [[sg.Text("1: OK | ", font="Courier"),
          sg.Text("2: Thank you very much | ", font="Courier"),
          sg.Text("3: Nice to meet you", font="Courier")],
         [sg.Text("4: See you next time | ", font="Courier"),
          sg.Text("5: Please wait a minute", font="Courier")],
         ],
        [sg.Image(
            filename='',
            key='-IMAGE-',
            expand_x=True,
            expand_y=True,
            zoom=5,
            subsample=3
         )],
        [sg.Multiline(
            parse,
            key="-ML-",
            write_only=True,
            size=(80, 6),
            reroute_cprint=True,
            reroute_stdout=True,
            font="Courier 12"
        )],
    ]

    # 创建并显示窗口
    window = sg.Window(
        "Sign Language Recognition",
        layout,
        location=(20, 20),
        use_default_focus=False,
        return_keyboard_events=True
    )

    # 参数0表示笔记本电脑自带摄像头，如果使用外接摄像头需要更换为其他数字
    cap = cv2.VideoCapture(0)

    # 设置视频编码方式及视频帧宽度高度
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 设置视频帧率
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 字母表
    ALPHABET = {"0": "A", "1": "B", "2": "C", "3": "D",
                "4": "E", "5": "F", "6": "G", "7": "H",
                "8": "I", "10": "K", "11": "L", "12": "M",
                "13": "N", "14": "O", "15": "P", "16": "Q",
                "17": "R", "18": "S", "19": "T", "20": "U",
                "21": "V", "22": "W", "23": "X", "24": "Y"}
    print("[INFO] Load alphabet.")

    # 模型导入
    DEVICE = config.DEVICE
    model = ResNet18()
    optimizer = Adam(model.parameters())
    checkpoint = load(resource_path("checkpoints/resnet18_4.tar"))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    print("[INFO] Load model.")

    # 待输出字符串容器
    WORDS = []

    while True:
        event, values = window.read(timeout=33)

        # 按下Exit后退出界面
        if event == "Exit" or event == sg.WIN_CLOSED or is_pressed('Esc'):
            cap.release()
            break

        _, frame = cap.read()

        # # 视频帧镜像翻转处理
        # frame = cv2.flip(frame, 1)

        # 若摄像头未正常打开，输出报错信息
        if not cap.isOpened():
            sg.cprint("[Error] Cannot open camera!", font="Courier 12")

        # 视频帧左上角作为检测区域，转换为低分辨率灰度图，并在图像窗口中标注出
        detect_zone = frame[:200, :200]
        low_res = cv2.resize(
            detect_zone, (28, 28), interpolation=cv2.INTER_LINEAR
        )
        low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2GRAY)

        # 图像转为张量传入模型
        img = Tensor(low_res).to(DEVICE)
        img = img[None, None, :, :]
        img /= 255.0
        out = model(img)
        pred = out.max(1, keepdim=True)[1].item()
        prob, label = topk(out, 1)
        if float(prob) < 0.6:
            text = ""
        else:
            text = "{}: {:.3f}%".format(ALPHABET[str(pred)], float(prob) * 100)

        # 显示预测结果
        font = cv2.FONT_HERSHEY_SIMPLEX
        frame = cv2.putText(frame, text, (10, 240), font, 1, (0, 255, 255), 4)

        # 绘制检测区域外框
        cv2.rectangle(frame, (0, 0), (200, 200), (0, 255, 255), 2)

        # PySimpleGUI不支持JPG图片，需要先转换为PNG图片并转为字节
        imgbytes = cv2.imencode('.png', frame)[1].tobytes()

        # 显示图像
        window["-IMAGE-"].update(data=imgbytes)

        # 保存图像
        if event == "Save" or is_pressed("s"):
            filename = datetime.datetime.now()
            filename = filename.strftime("%Y%m%d_%H%M%S")
            FOLDER_PATH = "D:/img/"
            if not os.path.exists(FOLDER_PATH):
                os.makedirs(FOLDER_PATH)
            fn = FOLDER_PATH + str(filename) + ".jpg"
            success = cv2.imwrite(fn, frame)
            if success:
                sg.cprint(
                    "[INFO] Save image at {}.".format(fn)
                )

        # 添加字母
        if event == "Add" or is_pressed("Enter"):
            WORDS.append(ALPHABET[str(pred)])
            ch = WORDS[-1]
            words = "".join(WORDS)
            sg.cprint(
                "[INFO] Add '{}'. Present words: {}.".format(ch, words),
                font="Courier 12"
            )

        # 删除字母
        if event == "Delete" or is_pressed("Backspace"):
            if WORDS:
                ch = WORDS.pop()
                words = "".join(WORDS)
                sg.cprint(
                    "[INFO] Delete '{}'. Remain words: {}.".format(ch, words),
                    font="Courier 12"
                )
            else:
                sg.cprint(
                    "[ERROR] List is empty! Please add letter first."
                )

        # 分词
        if event == "Split" or is_pressed("space"):
            WORDS.append(" ")
            sg.cprint("[INFO] Add space to words.", font="Courier 12")

        # 朗读字符串
        if event == "Speak" or is_pressed("Tab"):
            engine = pyttsx3.init()
            engine.say("".join(WORDS))
            engine.runAndWait()

        # 快捷回复1
        if is_pressed('1'):
            engine = pyttsx3.init()
            engine.say("OK")
            engine.runAndWait()

        # 快捷回复2
        if is_pressed('2'):
            engine = pyttsx3.init()
            engine.say("Thank you very much")
            engine.runAndWait()

        # 快捷回复3
        if is_pressed('3'):
            engine = pyttsx3.init()
            engine.say("Nice to meet you")
            engine.runAndWait()

        # 快捷回复4
        if is_pressed('4'):
            engine = pyttsx3.init()
            engine.say("See you next time")
            engine.runAndWait()

        # 快捷回复5
        if is_pressed('5'):
            engine = pyttsx3.init()
            engine.say("Please wait a minute")
            engine.runAndWait()

    window.close()


if __name__ == "__main__":
    main()
