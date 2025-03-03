Image_Analysis with QT (網點產生器/與強度分析（自動區分灰階強度）)
-

# 編譯與執行

Complie
-

qmake Image_Analysis.pro -o Makefile

make

Run
-

./Image_Analysis.app/Contents/MacOS/Image_Analysis



# v1.1 2025.03.01 更新

1. 修改部分介面與調整(QGraphicView的設定、輸出圖的原始用cv::COLOR_BGR2GRAY會有問題，改為自己的Image_Gray，輸出為RGB通道的灰階）。
2. 進行閥值調整，可以正常顯示，並把原本彩色漸層效果加回去。（主因是QGraphicView的設定，GRAY不能吃彩色）。
3. 加入processBar，但效果不如預期，先暫時放著。
4. 調整view2輸出，改成sub_function。

![介面](https://github.com/JIK-JHONG/practise/blob/main/Image_Analysis/demo2.jpeg)


# v1.0 2025.03.01 初版

熟悉QT creator開發環境，與GUI概念（類似tkiner）；

這邊主要實作

1. UI 介面
2. 監聽 / 點擊按鈕執行
3. include xxxxx.h (自己編寫的影像處理效果副程式)
4. 建立 icon (不確定是MacOS還是沒設定好，都沒有設定成功)
5. 熟悉  mainwindow.cpp / mainwindow.h / mainwindow.ui 作用。

![介面](https://github.com/JIK-JHONG/practise/blob/main/Image_Analysis/demo.jpeg)

