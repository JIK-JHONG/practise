#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>


using namespace cv;
using namespace std;

int save_div_image = 1 ;    // 儲存個別已辨識檔案，作為訓練用途。（ 0 = false , 1 = true）

// Point2f 相關應用：
// 定義點：         Point2f pt(10.5, 20.5); 
// 計算距離：       float dist = norm(pt1 - pt2);
// 用於影像變換：    Mat rotMat = getRotationMatrix2D(Point2f(100, 100), 30, 1.0);


// CV_8UC1	8-bit 單通道（灰階影像）
// CV_8UC3	8-bit 三通道（BGR 彩色影像）
// CV_8UC4	8-bit 四通道（BGRA，含透明度）
// CV_16UC1	16-bit 無符號整數灰階影像
// CV_16SC1	16-bit 有符號整數灰階影像
// CV_32FC1	32-bit 浮點數灰階影像
// CV_64FC1	64-bit 浮點數灰階影像


array<int, 3> image_info(const Mat &input){
    if (input.empty()) {
        cerr << "Error: Could not open image!" << endl;
        return {0, 0, 0};
    }
    array<int, 3> dim;
    dim[0] = input.rows;
    dim[1] = input.cols;
    dim[2] = input.channels();
    return dim;
}

void image_info_c(const Mat &input, int *dim){
    if (input.empty()) {
        cerr << "Error: Could not open image!" << endl;
        return ;
    }
    dim[0] = input.rows;
    dim[1] = input.cols;
    dim[2] = input.channels();
    string image_type;
    if (dim[2] == 1) {
        image_type = "GRAY";  // 單通道
    } else if (dim[2] == 3) {
        image_type = "BGR";   // OpenCV 預設 BGR
    } else if (dim[2] == 4) {
        image_type = "BGRA";  // 含透明通道
    } else {
        image_type = "Unknown";
    }

    // printf("Info Image W = %d , H = %d , C = %d\n",dim[0],dim[1],dim[2]);
    printf("---\n");
    printf("Info: Image W x H ( %s ) = ( %d x %d )\n", image_type.c_str(), dim[1], dim[0]);
    printf("---\n");
    
}


Mat load_image(const string &filepath){
    Mat image_read = imread(filepath, IMREAD_UNCHANGED);
    if (image_read.empty()) {
        cerr << "Error: Unable to open image " << filepath << endl;
    }
    return image_read;
}

void saveImage(const Mat &image, const string &outputFilename) {
    if (!imwrite(outputFilename, image)) {
        cerr << "Error: Could not save image to " << outputFilename << endl;
    }
}

Mat Image_Effect_01(const Mat &image,int factor=0){
    Mat copy_img = image.clone();
    return copy_img;

} 
Mat Image_replace_center(const Mat &soruce,Mat &target, int w, int h,int c){
    int center_w = w / 2 ;
    int center_h = h / 2 ;
    for (int k=0;k<c;k++){
        for (int j=0;j<h;j++){
            for (int i=0;i<w;i++){                
                target.at<Vec3b>(i+center_w, j+center_h)[k] = soruce.at<Vec3b>(i, j)[k];
            }
        }
    }
    return target;

}

Mat ImageBlur(const Mat& input, Point start = Point(0, 0), Size area = Size(), Size blur_size = Size(50, 50)) {
    Mat output = input.clone();
    if (area.width == 0 || area.height == 0) {
        blur(output, output, blur_size);
    } else {
        Rect roi(start, area);
        Mat blurred_roi;
        blur(output(roi), blurred_roi, blur_size);
        blurred_roi.copyTo(output(roi));
    }
    return output;
}



Mat ImageGray(const Mat& input, int option = 1) {
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();
    Mat gray(height, width, CV_8UC3, Scalar(255, 255, 255));
    int max_color = 1;
    int min_color = 255 ;
    // if (option == 1){

    // }else{
    for (int j=0;j<height;j++){
        for (int i=0;i<width;i++){
            // int tmp_balance =  ( input.at<Vec3b>(j, i)[0] + input.at<Vec3b>(j, i)[1] + input.at<Vec3b>(j, i)[2] ) / 3;
            int tmp_balance =  round(input.at<Vec3b>(j, i)[0] * 0.299 + input.at<Vec3b>(j, i)[1] * 0.587 + input.at<Vec3b>(j, i)[2] *0.114);
            if (tmp_balance < min_color){
                min_color = tmp_balance;
            }
            if (tmp_balance > max_color){
                max_color = tmp_balance;
            }

            // for (int k=0;k<channel;k++){
            //     gray.at<Vec3b>(j, i)[k] = tmp_balance;
            // }
        }
    }
    double factor_max = 255. / max_color;
    printf("Max_Fac = %f\n",factor_max);
    for (int j=0;j<height;j++){
        for (int i=0;i<width;i++){
            // int tmp_balance =  ( input.at<Vec3b>(j, i)[0] + input.at<Vec3b>(j, i)[1] + input.at<Vec3b>(j, i)[2] ) / 3;
            int tmp_balance =  round(input.at<Vec3b>(j, i)[0] * 0.299 + input.at<Vec3b>(j, i)[1] * 0.587 + input.at<Vec3b>(j, i)[2] *0.114);
            for (int k=0;k<channel;k++){
                int tmp = round (tmp_balance * factor_max );
                if (tmp >255){
                    tmp = 255 ;
                }
                gray.at<Vec3b>(j, i)[k] = tmp;
            }
        }
    }
    // }
    return gray;
}

Mat ImageBinary(const Mat& input, const std::string& mode_type = "normal",int low_threshold_force=127) {
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();
    printf("C = %d",channel);
    Mat output(height, width, CV_8UC3, Scalar(255, 255, 255));

    int low_threshold = low_threshold_force;
    int high_threshold = 255;

    if (mode_type == "normal"){
        for (int j=0;j<height;j++){
            for (int i=0;i<width;i++){
                int tmp_balance =  ( input.at<Vec3b>(j, i)[0] + input.at<Vec3b>(j, i)[1] + input.at<Vec3b>(j, i)[2] ) / 3;
                for (int k=0;k<channel;k++){                
                    if (tmp_balance >low_threshold){
                        tmp_balance = 255 ;
                    }else{
                        tmp_balance = 0;
                    }
                    output.at<Vec3b>(j, i)[k] = tmp_balance;
                }
            }
        }
    }else if (mode_type == "inverted") {
        for (int j=0;j<height;j++){
            for (int i=0;i<width;i++){
                int tmp_balance =  ( input.at<Vec3b>(j, i)[0] + input.at<Vec3b>(j, i)[1] + input.at<Vec3b>(j, i)[2] ) / 3;
                for (int k=0;k<channel;k++){      
                    int tmp = tmp_balance ;          
                    if (tmp_balance >low_threshold){
                        tmp = 0 ;
                    }else{
                        tmp = 255;
                    }
                    output.at<Vec3b>(j, i)[k] = tmp;
                    
                }
            }
        }
    }else if (mode_type == "enhance") {
        // int mid_threshold = (low_threshold + high_threshold) / 2 ;
        for (int j=0;j<height;j++){
            for (int i=0;i<width;i++){
                int tmp_balance =  ( input.at<Vec3b>(j, i)[0] + input.at<Vec3b>(j, i)[1] + input.at<Vec3b>(j, i)[2] ) / 3;
                for (int k=0;k<channel;k++){    
                    int tmp = tmp_balance ;                      
                    if (tmp_balance >=0 && tmp_balance < 85){
                        tmp = 0 ;
                    }else if (tmp_balance >=85 && tmp_balance < 170){
                        tmp = 170;
                    }else{
                        tmp = 255;
                    }
                    output.at<Vec3b>(j, i)[k] = tmp;
                }
            }
        }
    }

    return output;
}

Mat Image_Pick_Color(const Mat &input, string option="red",int force_use = 0 ,int low_threshold_set = 60){
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();
    int channel_set = 2 ;
    int low_threshold = low_threshold_set;
    Mat output(height,width, CV_8UC3, Scalar(0, 0, 0));

    if (option == "red"){
        channel_set = 2; 
    }else if (option == "green"){
        channel_set = 1; 
    }else if (option == "blue"){
        channel_set = 0; 
    }
    for (int j=0;j<height;j++){
        for (int i=0;i<width;i++){
            if (force_use == 1){
                if (input.at<Vec3b>(j, i)[channel_set] > low_threshold){
                    output.at<Vec3b>(j, i)[channel_set] = 255;     
                    // output.at<Vec3b>(j, i)[channel_set] = input.at<Vec3b>(j, i)[channel_set];           
                }else{
                    output.at<Vec3b>(j, i)[channel_set] = input.at<Vec3b>(j, i)[channel_set];               
                }
            }else{
                output.at<Vec3b>(j, i)[channel_set] = input.at<Vec3b>(j, i)[channel_set];      
            }
            
        }
    }
    return output;

    

}


Mat Image_Add_Layer(const Mat &input,const Mat &input2, int fac=1, int fac2=1){
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();
    Mat output(height,width, CV_8UC3, Scalar(0, 0, 0));

    for (int j=0;j<height;j++){
        for (int i=0;i<width;i++){
            for (int k=0;k<channel;k++){
                output.at<Vec3b>(j, i)[k] = int((input.at<Vec3b>(j, i)[k] * fac + input2.at<Vec3b>(j, i)[k] * fac2)/(fac + fac2)) ;   
            }     
        }
    }
    return output;

}

Mat Image_Add_Layer_3c(const Mat &input,const Mat &input2,const Mat &input3, int fac=1, int fac2=1, int fac3=1){
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();
    Mat output(height,width, CV_8UC3, Scalar(0, 0, 0));

    for (int j=0;j<height;j++){
        for (int i=0;i<width;i++){
            for (int k=0;k<channel;k++){
                output.at<Vec3b>(j, i)[k] = int((input.at<Vec3b>(j, i)[k] * fac + input2.at<Vec3b>(j, i)[k] * fac2 + input3.at<Vec3b>(j, i)[k] * fac3)) ;   
            }     
        }
    }
    return output;

}

// 影像平均化處理 (類似灰階化，但保留 BGR 格式)
Mat Image_Effect_02(const Mat &image) {
    int factor[] = {2,1,1};
    int factor_sum = factor[0] + factor[1] + factor[2];
    // 確保影像是 BGR 三通道
    if (image.channels() != 3) {
        return image.clone();  // 若非 BGR，直接回傳原圖
    }

    Mat copy_img = image.clone();
    int height = image.rows;  // 影像高度
    int width = image.cols;   // 影像寬度

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            if (j%4!=0 && i%4!=2){
                Vec3b pixel = image.at<Vec3b>(j, i);  // 讀取 BGR 像素
                int tmp_balance = (pixel[0] * factor[0] + pixel[1] * factor[1] + pixel[2] * factor[2]) / factor_sum; // 計算平均值

                Vec3b &new_pixel = copy_img.at<Vec3b>(j, i);  // 取得可修改的參考
                new_pixel[0] = tmp_balance; // 設定 B 通道
                new_pixel[1] = tmp_balance; // 設定 G 通道
                new_pixel[2] = tmp_balance; // 設定 R 通道
            }else{
                copy_img.at<Vec3b>(j, i)[0] =  image.at<Vec3b>(j, i)[0];
                copy_img.at<Vec3b>(j, i)[1] =  image.at<Vec3b>(j, i)[1];
                copy_img.at<Vec3b>(j, i)[2] =  image.at<Vec3b>(j, i)[2];
            }
        }
    }
    return copy_img;
}
Mat Image_Effect_03(const Mat &input,int shift_unit=5){
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();
    Mat output_r = Image_Pick_Color(input,"red");
    Mat output_g = Image_Pick_Color(input,"green");
    Mat output_b = Image_Pick_Color(input,"blue");
    
    for (int j=0;j<height-shift_unit;j++){
        for (int i=0;i<width-shift_unit;i++){
            for (int k=0;k<channel;k++){
                output_g.at<Vec3b>(j, i)[k] =  output_g.at<Vec3b>(j+shift_unit, i+shift_unit)[k];
            }     
        }
    }
    for (int j=0;j<height;j++){
        for (int i=0;i<width;i++){
            for (int k=0;k<channel;k++){
                output_r.at<Vec3b>(j, i)[k] =  output_r.at<Vec3b>(j, i)[k];
            }     
        }
    }
    for (int j=0;j<height-shift_unit*2;j++){
        for (int i=0;i<width-shift_unit*2;i++){
            for (int k=0;k<channel;k++){
                output_b.at<Vec3b>(j, i)[k] =  output_b.at<Vec3b>(j+shift_unit*2, i+shift_unit*2)[k];
            }     
        }
    }
    // output_r = Image_Add_Layer(output_r,output_g);
    // output_r = Image_Add_Layer(output_r,output_b);
    
    return Image_Add_Layer_3c(output_r,output_g,output_b);

}
Mat Image_Effect_04(const Mat &input,int contrast=200, int brightness=50){
    // (contrast/127 + 1) - contrast + brightness
    int low_threshold = 60;
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();
    
    double alpha = (contrast / 127.0) + 1.0;
    int beta = brightness;
    printf("Enhance_Bright = %d\n",int(alpha * 127 + beta));
    Mat output(height,width, CV_8UC3, Scalar(0, 0, 0));
    
    for (int j=0;j<height;j++){
        for (int i=0;i<width;i++){
            int tmp_balance = ((input.at<Vec3b>(j, i)[0] + input.at<Vec3b>(j, i)[1] + input.at<Vec3b>(j, i)[2]) / 3 );
            if ( tmp_balance > low_threshold){
                for (int k=0;k<channel;k++){
                    if (int(input.at<Vec3b>(j, i)[k] * alpha  + beta) > 255){
                        output.at<Vec3b>(j, i)[k] =  255;
                    }else{
                        output.at<Vec3b>(j, i)[k] =  int(input.at<Vec3b>(j, i)[k] * alpha  + beta);
                    }
                    
                }    
            }else{
                for (int k=0;k<channel;k++){
                    output.at<Vec3b>(j, i)[k] =  input.at<Vec3b>(j, i)[k];
                }   
            }
             
        }
    }
    return output;
}

Mat Pixelized(const Mat& input, int pixel_size=10){
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();
    int height_pixel = height / pixel_size ; 
    int width_pixel = width / pixel_size ; 
    int pixel_area = pixel_size * pixel_size ; 

    Mat output(height,width,CV_8UC3, Scalar(0, 0, 0));

    for (int j = 0; j < height_pixel+1; j++) {
        for (int i = 0; i < width_pixel+1; i++) {
            int sub_r = 0;
            int sub_g = 0;
            int sub_b = 0;
            for (int kj=0;kj<pixel_size;kj++){
                for (int ki=0;ki<pixel_size;ki++){                    
                    sub_r += input.at<Vec3b>(j*pixel_size + kj, i*pixel_size + ki)[2] ; 
                    sub_g += input.at<Vec3b>(j*pixel_size + kj, i*pixel_size + ki)[1] ; 
                    sub_b += input.at<Vec3b>(j*pixel_size + kj, i*pixel_size + ki)[0] ;                     
                }
            }
            sub_r/=pixel_area;
            sub_g/=pixel_area;
            sub_b/=pixel_area;
            sub_r = int(sub_r);
            sub_g = int(sub_g);
            sub_b = int(sub_b);

            for (int kj=0;kj<pixel_size;kj++){
                for (int ki=0;ki<pixel_size;ki++){                    
                    output.at<Vec3b>(j*pixel_size + kj, i*pixel_size + ki)[2] = sub_r;
                    output.at<Vec3b>(j*pixel_size + kj, i*pixel_size + ki)[1] = sub_g;
                    output.at<Vec3b>(j*pixel_size + kj, i*pixel_size + ki)[0] = sub_b;                 
                }
            }

        }
    }
    return output;

}

Mat Wave(const Mat& input, int amp = 30, double frq = 0.1, string direction = "hor") {
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();

    Mat output(height,width,CV_8UC3, Scalar(0, 0, 0));

    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int mod_i = i, mod_j = j;

            if (direction == "hor") {
                mod_i = i + int(amp * sin(j * frq));
            } else if (direction == "ver") {
                mod_j = j + int(amp * sin(i * frq));
            }

            // 限制範圍，避免超出影像邊界
            if (mod_i < 0 ){
                mod_i = 0 ;
            }else if (mod_i > width){
                mod_i = width;
            }
            if (mod_j < 0 ){
                mod_j = 0 ;
            }else if (mod_j > height){
                mod_j = height;
            }

            // 變形後的像素映射
            output.at<Vec3b>(j, i) = input.at<Vec3b>(mod_j, mod_i);
        }
    }
    return output;
}

Mat Diff_Image(const Mat& input,const int order=1){
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();
    Mat output(height,width,CV_8UC3, Scalar(0, 0, 0));
    for (int k = 0; k < channel; k++) {
        for (int j = 0; j < height - order; j++) {
            for (int i = 0; i < width - order; i++) {
                if (order == 1){
                    int tmp = input.at<Vec3b>(j+1, i+1)[k] - input.at<Vec3b>(j, i)[k];
                    if (tmp > 255){
                        tmp = 255;
                    }else if (tmp < 0 ){
                        tmp = 0;
                    }else{
                        tmp = tmp;
                    }
                    
                    output.at<Vec3b>(j, i)[k] = tmp ;
                }else if (order == 2){
                    int tmp = (input.at<Vec3b>(j+2, i+2)[k] - 2*input.at<Vec3b>(j+1, i+1)[k] +  input.at<Vec3b>(j, i)[k]) / 2;
                    if (tmp > 255){
                        tmp = 255;
                    }else if (tmp < 0 ){
                        tmp = 0;
                    }else{
                        tmp = tmp;
                    }
                    
                    output.at<Vec3b>(j, i)[k] = tmp;
                }
            }
        }
    }
    return output;

}


// 旋轉影像 N 度
Mat rotateImage(const Mat& src, double angle) {
    Point2f center(src.cols / 2.0, src.rows / 2.0);
    Mat rotMat = getRotationMatrix2D(center, angle, 1.0);
    
    Mat rotated;
    warpAffine(src, rotated, rotMat, src.size());
    return rotated;
}


void image_preProcess(Mat &input){
    int channel = input.channels();
    if (channel == 1){
        cvtColor(input, input, COLOR_GRAY2BGR);
        printf("image_input > to RGB\n");
    }else{
        printf("image_input > unchanged\n");
    }

}

void Compare_View4(const Mat &input, const Mat &input2, const Mat &input3, const Mat &input4){
    int height = input.rows;
    int width = input.cols;

    // 轉換成 BGR 以便顯示
    Mat output_tmp, output_tmp2, output_tmp3, output_tmp4;
    output_tmp = input.clone();
    output_tmp2 = input2.clone();
    output_tmp3 = input3.clone();
    output_tmp4 = input4.clone();

    image_preProcess(output_tmp);
    image_preProcess(output_tmp2);
    image_preProcess(output_tmp3);
    image_preProcess(output_tmp4);

    // 建立寬度加倍的影像來放置三張影像
    int width_compare = width * 4;
    Mat output(height, width_compare, CV_8UC3, Scalar(255, 255, 255));
    // 複製影像到 output
    output_tmp.copyTo(output(Rect(0, 0, width, height)));         // 左邊
    output_tmp2.copyTo(output(Rect(width, 0, width, height)));    // 中間
    output_tmp3.copyTo(output(Rect(width * 2, 0, width, height))); // 右邊
    output_tmp4.copyTo(output(Rect(width * 3, 0, width, height))); // 右邊
    saveImage(output,"image_compare_set.jpeg");
    imshow("Output", output);
    waitKey(0);
}


void show_image_rgb(const Mat& input){
    int height = input.rows;
    int width = input.cols;
    int channel = input.channels();
    int total_mesh = height * width ; 
    double sum_r = 0 ;
    double sum_g = 0 ;
    double sum_b = 0 ;

    for (int j=0;j<height;j++){
        for (int i=0;i<width;i++){
            sum_b += double(input.at<Vec3b>(j, i)[0]) / 255.;
            sum_g += double(input.at<Vec3b>(j, i)[1]) / 255.;
            sum_r += double(input.at<Vec3b>(j, i)[2]) / 255.;
        }
    }

    printf("---\n");
    printf("Ave : R = %d , G = %d , B = %d\n", int(sum_r/total_mesh*255), int(sum_g/total_mesh*255), int(sum_b/total_mesh*255));
    printf("---\n");


}




void init_rgba2rgb(Mat& input){
    cvtColor(input, input, cv::COLOR_BGRA2BGR);
}

int main(){
    string file_name = "demo_bird_img.jpeg";    
    Mat image = load_image(file_name); 
    int *array_info = (int*) malloc(sizeof(int) * 3);
    if (array_info == nullptr) {
        cerr << "Memory allocation failed!" << endl;
        return -1;  // 退出程序
    }

    if (image.channels()==4){
        printf("為RGBA圖檔類型，進行轉換為RGB\n");
        init_rgba2rgb(image);
    }
    


    image_info_c(image, array_info);
    printf("Image W = %d , H = %d , C = %d\n",array_info[0],array_info[1],array_info[2]);

    Mat copy_set = image.clone();
    show_image_rgb(copy_set);


    Mat img_with_contours_b ;
    cvtColor(image,img_with_contours_b,COLOR_BGR2GRAY);
    // img_with_contours_b = ImageBlur(img_with_contours_b,Point(0, 0),Size(),Size(4, 4));
    int kernel_size = 4 ;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size)); // 5x5 矩形結構元素
    cv::morphologyEx(img_with_contours_b, img_with_contours_b, cv::MORPH_OPEN, kernel);
    
    // Image_Effect_04(img_with_contours_b);
    threshold(img_with_contours_b,img_with_contours_b,90,255,THRESH_BINARY);

    // cv::imshow("Image img_with_contours - pre", img_with_contours_b);
    // cv::waitKey(0);  
    std::vector<std::vector<Point>> contours;
    findContours(img_with_contours_b, contours, RETR_TREE, CHAIN_APPROX_NONE);
    Mat img_with_contours = image.clone();
    // for (size_t i = 1; i < contours.size(); i++) {
    //     drawContours(img_with_contours, contours, i, Scalar(0, 255, 0), 4); // 畫出綠色輪廓   
    // }

    // drawContours(img_with_contours, contours, -1, Scalar(0, 255, 0), 4); // 畫出綠色輪廓   
    // cv::imshow("Image img_with_contours", img_with_contours);
    // cv::waitKey(0);    


    // 建立遮罩影像
    Mat mask_image(image.size(), CV_8U, Scalar(0));

    // 繪製所有輪廓
    for (size_t i = 1; i < contours.size(); i++) {
        drawContours(mask_image, contours, i, Scalar(255), FILLED);
    }

    // 建立結果影像，只保留輪廓內的內容
    Mat result(image.size(),CV_8UC3, Scalar(255, 255, 255));
    image.copyTo(result, mask_image);


    for (size_t i = 1; i < contours.size(); i++) {
        int shift_mesh = 10 ;
        Rect rect = boundingRect(contours[i]);  // 找出輪廓的外接矩形
        if (save_div_image == 1){
            Mat cropped = image(rect);  // 從原圖裁剪出該區域
            imwrite("cropped_contour_" + to_string(i) + ".jpeg", cropped);
        }
        string text = to_string(i);
        Point textPos(rect.x , rect.y - shift_mesh);  // 文字的位置
        int font = FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.8;
        Scalar textColor(255, 0, 0);  // 文字顏色 (白色)
        int thickness = 2;
        putText(result, text, textPos, font, fontScale, textColor, thickness);
    }

    Compare_View4(image, img_with_contours, mask_image, result);
    

}