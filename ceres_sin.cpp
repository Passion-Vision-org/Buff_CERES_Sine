//#include "ceres_sin.h"
/*库包含
时间库：<ctime>
优化库：<ceres/ceres.h>
opencv库：<opencv2/opencv.hpp>
矩阵库：<Eigen/Core>
文件编写库：<yaml-cpp/yaml.h>
绘图库：<matplotlibcpp.h>
*/
#include<iostream>
#include<ctime>
#include<vector>
#include <deque>
#include<future>
#include <random>

//#include <debug.h>
#include <ceres/ceres.h>
#include <Eigen/Core>
#include <matplotlibcpp.h>//已经配置完毕
#include <opencv2/opencv.hpp>

//#include <yaml-cpp/yaml.h>

#define DRAW_PREDICT
/*命名空间*/
namespace plt = matplotlibcpp;

using namespace std;
using namespace cv;
using namespace plt;

// const string pf_path = "filter_param.yaml";//粒子滤波参数文件

// class ParticleFilter//粒子滤波类
// {
// public:
//     //ParticleFilter(YAML::Node &config,const string param_name);
//     //ParticleFilter();
//     //~ParticleFilter();

//     Eigen::VectorXd predict();
//     bool initParam(YAML::Node &config,const string param_name);
//     bool initParam(ParticleFilter parent);
//     bool update(Eigen::VectorXd measure);
//     bool is_ready;
// private:

//     bool resample();

//     int vector_len;
//     int num_particle;

//     Eigen::MatrixXd process_noise_cov;
//     Eigen::MatrixXd observe_noise_cov;
//     Eigen::MatrixXd weights;

//     Eigen::MatrixXd matrix_estimate;
//     Eigen::MatrixXd matrix_particle;
//     Eigen::MatrixXd matrix_weights;

// };


/*创建预测类用于使用:在测试过程中出现了大量的类中的结构体成员未定义的问题*/

        


    struct CURVE_FITTING_COST       //代价函数计算模型
    {
        CURVE_FITTING_COST (double x, double y) : _x ( x ), _y ( y ) {}     //x和y都是之后的输入量
        // 残差的计算
        template <typename T>
        bool operator() (
            const T* params,     //params存储4个待拟合参数，其中初相角之后会进行单独的二次拟合
            T* residual) const     // 残差
        {
            residual[0] = T (_y) - params[0] * ceres::sin(params[1] * T (_x) + params[2]) - params[3]; // f(x) = a * sin(ω * t + θ) + b，残差表达式
            return true;
        }
        const double _x, _y;    // x,y数据，为输入量

    };

    struct CURVE_FITTING_COST_PHASE     //代价函数模型，用于拟合初相角
    {
        CURVE_FITTING_COST_PHASE (double x, double y, double a, double omega, double dc) : _x (x), _y (y), _a(a), _omega(omega), _dc(dc){}
        // 残差的计算
        template <typename T>
        bool operator() (
        const T* phase,     // 模型参数，有1维
        T* residual) const     // 残差
        {
            residual[0] = T (_y) - T (_a) * ceres::sin(T(_omega) * T (_x) + phase[0]) - T(_dc); // f(x) = a * sin(ω * t + θ)+b
            return true;
        }
        const double _x, _y, _a, _omega, _dc;    // x,y数据点以及第一次拟合得到的另外三个参数
    };

    //待打击扇页信息：转速、距离、时间戳
    struct TargetInfo
    {
        double speed;
        //double dist;
        double timestamp;
    };

    //关键变量定义和初始化
    double params[4];
    double bullet_speed = 28;                                                //用于计算弹丸飞行时间同时用于之后的弹速修改
    
    std::deque<TargetInfo> history_info;                                     //目标队列，放入待打击扇叶信息
    const int max_timespan = 20000;                                          //最大时间跨度，大于该时间重置预测器(ms)
    const double max_rmse = 0.4;                                             //TODO:回归函数最大Cost，如果拟合后误差超过它则重新预测
    const int max_v = 3;                                                     //设置最大速度,单位rad/s
    const int max_a = 8;                                                     //设置最大角加速度,单位rad/s^2
    const int history_deque_len_cos = 250;                                   //大符全部参数拟合队列长度
    const int history_deque_len_phase = 100;                                 //大符相位参数拟合队列长度
    const int history_deque_len_uniform = 100;                               //小符转速求解队列长度
    
    const int delay_small = 175;                                             //小符发弹延迟
    const int delay_big = 100;                                               //大符发弹延迟
    const int window_size = 2;   

    TargetInfo last_target;                                                  //最后目标//存疑
    // ParticleFilter pf;
    // ParticleFilter pf_param_loader;
    TargetInfo target;//确实是结构体型的
    int mode=1;                                                                //预测器模式，0为小符，1为大符
    int last_mode;
    bool is_params_confirmed;

    bool predict(double speed, double dist, int timestamp, double &result);
    double calcAimingAngleOffset(double params[4], double t0, double t1, int mode);
    double shiftWindowFilter(int start_idx);
    bool setBulletSpeed(double speed);
    double evalMAPE(double params[4]);
    double evalRMSE(double params[4]);





/*拟合参量误差计算函数（均方根误差），对于拟合参量是否满足要求作为筛选标准*/
double evalRMSE(double params[4])
{
    double rmse_sum = 0;
    double rmse = 0;
    for (auto target_info : history_info)
    {
        auto t = (float)(target_info.timestamp) / 1e3;
        auto pred = params[0] * sin (params[1] * t + params[2]) + params[3];//pred是预测过后的速度值
        auto measure = target_info.speed;//measure是实际得到的速度值
        rmse_sum+=pow((pred - measure),2);//实际和预测的差值进行平方再求和
    }
    rmse = sqrt(rmse_sum / history_info.size());//差值平方求和之后除以数据总长度再开方就是均方根误差
    return rmse;
}

// template<typename T>
// bool initMatrix(Eigen::MatrixXd &matrix,std::vector<T> &vector)
// {
//     int cnt = 0;
//     for(int row = 0;row < matrix.rows();row++)
//     {
//         for(int col = 0;col < matrix.cols();col++)
//         {
//             matrix(row,col) = vector[cnt];
//             cnt++;
//         }
//     }
//     return true;
// }
// /*生成正态分布矩阵*/
// bool randomlizedGaussianColwise(Eigen::MatrixXd &matrix, Eigen::MatrixXd &cov)
// {
//     std::random_device rd;
//     default_random_engine e(rd());
//     std::vector<normal_distribution<double>> normal_distribution_list;

//     //假设各个变量不相关
//     for (int i = 0; i < cov.cols(); i++)
//     {
//         normal_distribution<double> n(0,cov(i,i));
//         normal_distribution_list.push_back(n);
//     }


//     for (int col = 0; col < matrix.cols(); col++)
//     {
//         // cout<<normal_distribution_list[col](e)<<endl;
//         for(int row = 0; row < matrix.rows(); row++)
//         {
//             auto tmp = normal_distribution_list[col](e);
//             matrix(row, col) = tmp;
//             // matrix(row,col) = 1;
//         }
//     }

//     return true;
// }
// /*粒子滤波函数一：原理存疑*/
// bool ParticleFilter::initParam(YAML::Node &config,const string param_name)
// {
//     //初始化向量长度与粒子数
//     vector_len = config[param_name]["vector_len"].as<int>();
//     num_particle = config[param_name]["num_particle"].as<int>();
//     Eigen::MatrixXd process_noise_cov_tmp(vector_len,vector_len);
//     Eigen::MatrixXd observe_noise_cov_tmp(vector_len,vector_len);
//     //初始化过程噪声矩阵
//     auto read_vector = config[param_name]["process_noise"].as<vector<float>>();
//     initMatrix(process_noise_cov_tmp,read_vector);
//     process_noise_cov = process_noise_cov_tmp;
//     //初始化观测噪声矩阵
//     read_vector = config[param_name]["observe_noise"].as<vector<float>>();
//     initMatrix(observe_noise_cov_tmp,read_vector);
//     observe_noise_cov = observe_noise_cov_tmp;
//     //初始化粒子矩阵及粒子权重
//     // matrix_particle = 3 * Eigen::MatrixXd::Random(num_particle, vector_len);
//     matrix_particle = Eigen::MatrixXd::Zero(num_particle, vector_len);
//     randomlizedGaussianColwise(matrix_particle, process_noise_cov);
//     matrix_weights = Eigen::MatrixXd::Ones(num_particle, 1) / float(num_particle);
//     is_ready = false;
    
//     return true;
// }
// /*粒子滤波（与卡尔曼滤波进行比较：粒子滤波是无参滤波，基于蒙特卡罗数据抽样，样本被称为粒子）函数二*/
// bool ParticleFilter::initParam(ParticleFilter parent)
// {
//     vector_len = parent.vector_len;
//     num_particle = parent.num_particle;
//     process_noise_cov = parent.process_noise_cov;
//     observe_noise_cov = parent.observe_noise_cov;
//     //初始化粒子矩阵及粒子权重
//     matrix_particle = Eigen::MatrixXd::Zero(num_particle, vector_len);
//     randomlizedGaussianColwise(matrix_particle, process_noise_cov);
//     matrix_particle = 3 * Eigen::MatrixXd::Random(num_particle, vector_len);
//     matrix_weights = Eigen::MatrixXd::Ones(num_particle, 1) / float(num_particle);
//     is_ready = false;

//     return true;
// }
// /*粒子滤波更新函数*/
// bool ParticleFilter::update(Eigen::VectorXd measure)
// {
//     Eigen::MatrixXd gaussian = Eigen::MatrixXd::Zero(num_particle, vector_len);
//     Eigen::MatrixXd mat_measure = measure.replicate(1,num_particle).transpose();
//     auto err = ((measure - (matrix_particle.transpose() * matrix_weights)).norm());
//     // cout<<num_particle<<" err "<<err<<endl;

//     if (is_ready)
//     {
//         //序列重要性采样
//         matrix_weights = Eigen::MatrixXd::Ones(num_particle, 1);
//         //按照高斯分布概率密度函数曲线右半侧计算粒子权重
//         for(int i = 0; i < matrix_particle.cols(); i++)
//         {
//             auto sigma = observe_noise_cov(i,i);
//             Eigen::MatrixXd weights_dist = (matrix_particle.col(i) - mat_measure.col(i)).rowwise().squaredNorm();
//             Eigen::MatrixXd tmp = ((-(weights_dist / pow(sigma, 2)) / matrix_particle.cols()).array().exp() / (sqrt(CV_2PI) * sigma)).array();
//             matrix_weights = matrix_weights.array() * tmp.array();
//         }
//         matrix_weights /= matrix_weights.sum();
//         double n_eff = 1.0 / (matrix_weights.transpose() * matrix_weights).value();
//         //TODO:有效粒子数少于一定值时进行重采样,该值需在实际调试过程中修改
//         // if (n_eff < 0.5 * num_particle)
//         if (err > observe_noise_cov(0,0) || (n_eff < 0.5 * num_particle))
//         {
//             // cout<<"res"<<num_particle<<endl;
//             resample();
//         }
//     }
//     else
//     {
//         matrix_particle+=mat_measure;
//         is_ready = true;
//         return false;
//     }
//     return true;
// }
// /*粒子滤波下的预测*/
// Eigen::VectorXd ParticleFilter::predict()
// {
//     Eigen::VectorXd particles_weighted = matrix_particle.transpose() * matrix_weights;
//     return particles_weighted;
// }
// /*粒子滤波下的重采样*/
// bool ParticleFilter::resample()
// {    
    
//     //重采样采用低方差采样,复杂度为O(N),较轮盘法的O(NlogN)更小,实现可参考<Probablistic Robotics>
//     std::random_device rd;
//     default_random_engine e(rd());
//     std::uniform_real_distribution<> random {0.0, 1.d / num_particle};

//     int i = 0;
//     double c = matrix_weights(0,0);
//     auto r = random(e);
//     Eigen::MatrixXd matrix_particle_tmp = matrix_particle;

//     for (int m = 0; m < num_particle; m++)
//     {
//         auto u = r + m * (1.d / num_particle);
//         // 当 u > c 不进行采样
//         while (u > c)
//         {
//             i++;
//             c = c + matrix_weights(i,0);
//         }
//         matrix_particle_tmp.row(m) = matrix_particle.row(i);
//     }
//     Eigen::MatrixXd gaussian = Eigen::MatrixXd::Zero(num_particle, vector_len);
//     randomlizedGaussianColwise(gaussian, process_noise_cov);
//     matrix_particle = matrix_particle_tmp + gaussian;
//     matrix_weights = Eigen::MatrixXd::Ones(num_particle, 1) / float(num_particle);
//     return true;
// }





   


int main()
{
    is_params_confirmed = false;
    params[0] = 0;
    params[1] = 0; 
    params[2] = 0; 
    params[3] = 0;
    // YAML::Node config = YAML::LoadFile(pf_path);//////////将yaml文件中的变量放在config中便于粒子滤波使用数据
    // pf_param_loader.initParam(config, "buff");//////////////


    for(int i=1;i<200;i++)//为什么数据点影响这么大
    {   double ar = 2.0,br=3.0,cr=2.0,dr=3.0;
        double w_sigma = 0.01;//噪声的sigma值
        cv::RNG rng;//opencv随机数产生器,没有发挥作用
        vector<double> y_data,x_data;
        double x = i/20.0 ;
		x_data.push_back(x);
        //double t=rng.gaussian(w_sigma);
        double noise_y=ar*sin(br*x+cr)+dr+rng.gaussian(w_sigma);
        //cout<<rng.gaussian(w_sigma)<<" "<<noise_y<<endl;
        y_data.push_back(ar*ceres::sin(br*x+cr)+dr + rng.gaussian(w_sigma));
        target.speed=y_data.back();//不断把尾部的数据放入目标队列中
        target.timestamp=x_data.back();
        //cout<<rng.gaussian(w_sigma*w_sigma)<<endl;
        //cout<<target.timestamp<<" "<<x_data.back()<<" "<<target.speed<<" "<<y_data.back()<<endl;数据是一样的，没有问题;y值计算也是正确的没有问题
 /*对数据（对速度即y值）使用粒子滤波，没有使用滑窗滤波*/
    // auto is_ready = pf.is_ready;//先赋值为初始默认值false，滤波完成后赋值为ture
    // Eigen::VectorXd measure(1);
    // measure<<target.speed;
    // pf.update(measure);//使用定义的update函数进行粒子速度滤波


    //  if (is_ready)
    //  {
    //      auto predict = pf.predict();
    //      // cout<<predict<<" : "<<speed<<endl;
    //      target.speed = predict[0];
    //  }
/*进行大小符模式以及一次变量是否拟合完成的判断对信息队列的长度;根据信息队列的长度是否达到设定的长度来进行放入和弹出数据*/
    auto deque_len = 0;
    if (mode == 0)
    {
        deque_len = history_deque_len_uniform;
    }
    else if (mode == 1)
    {
        if (!is_params_confirmed)
            deque_len = history_deque_len_cos;
        else
            deque_len = history_deque_len_phase;
    }
    if (history_info.size() < deque_len)    
    {
        history_info.push_back(target);
        //last_target = target;
        //return false;//让它不运行的是这个位置，这个位置直接让主函数返回了

    }
    else if (history_info.size() == deque_len)
    {
        history_info.pop_front();
        history_info.push_back(target);
    }
    else if (history_info.size() > deque_len)
    {
        while(history_info.size() >= deque_len)
            history_info.pop_front();
        history_info.push_back(target);
    }
    }
 /*计算扇叶旋转方向（因为速度正铉函数是恒为正或恒为负），给之后拟合数据的使用做准备*/
    double rotate_speed_sum = 0;
    int rotate_sign=1;
    for (auto target_info : history_info) //遍历队列累加速度值
        rotate_speed_sum += target_info.speed;
    auto mean_velocity = rotate_speed_sum / history_info.size();
    
    for (auto target_info : history_info)
        cout<<target_info.timestamp<<" "<<target_info.speed<<endl;

 //cout<<"测试位置一hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh"<<endl;
 //一次拟合未进行或未成功下的进行一次拟合
        while (!is_params_confirmed)
        {
            ceres::Problem problem;               //构建问题
            ceres::Solver::Options options;       //构建选项
            ceres::Solver::Summary summary;       //优化信息
            double params_fitting[4] = {1, 1, 1, mean_velocity};//初始化待拟合参数：但是前三个值给的是中间值，最后值给的是计算值，减小与真实值的初始差，降低迭代时间
        //将旋转方向变成影响拟合的参数值，顺时针为-1
        // if (rotate_speed_sum / fabs(rotate_speed_sum) >= 0)
        //         rotate_sign = 1;
        // else
        //         rotate_sign = -1;
        //添加误差项并自动求导，将初始化后的params_fitting进行拟合
        for (auto target_info : history_info)
        {
            problem.AddResidualBlock (     // 向问题中添加误差项
            // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
                new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 4> ( 
                        new CURVE_FITTING_COST ((float)(target_info.timestamp) ,target_info.speed  * rotate_sign)//这里对速度乘上旋转标志，让速度量变成恒正的量，相当与把图像往坐标轴上翻过去了
                        ),                              //添加在误差项里面的俩个参数分别是时间和速度，也就是最开始的x和y输入值
                new ceres::CauchyLoss(0.5),     //柯西核函数
                params_fitting                  // 待估计参数
                );
        }
        cout<<params_fitting[0]<<" "<<params_fitting[1]<<" "<<params_fitting[2]<<" "<<params_fitting[3]<<endl;//是正常值
        /*配置求解器:后来添加*/
        options.linear_solver_type=ceres::DENSE_QR;
        options.minimizer_progress_to_stdout=true;
        options.max_num_iterations=200;
        options.num_threads=1;
        //设置待拟合4个参数的上下限，之后便可进行solve（但是拟合出来的结果依旧不堪）
        problem.SetParameterLowerBound(params_fitting,0,0);
        problem.SetParameterUpperBound(params_fitting,0,10);//较大的上下限用于观察拟合效果，用于测试
        problem.SetParameterLowerBound(params_fitting,1,0);
        problem.SetParameterUpperBound(params_fitting,1,10);
        problem.SetParameterLowerBound(params_fitting,2,-CV_PI);
        problem.SetParameterUpperBound(params_fitting,2,CV_PI);
        problem.SetParameterLowerBound(params_fitting,3,0);
        problem.SetParameterUpperBound(params_fitting,3,10);
        //拟合求解并将信息放入summary中
        ceres::Solve(options, &problem, &summary);
        cout<<params_fitting[0]<<" "<<params_fitting[1]<<" "<<params_fitting[2]<<" "<<params_fitting[3]<<endl;//一次拟合之后得到不理想值
        //拟合得到的参数数据放入临时数组用于判断是否有效
        double params_tmp[4] = {params_fitting[0] * rotate_sign, params_fitting[1], params_fitting[2], params_fitting[3] * rotate_sign};//把拟合得到的速度量再次取反回归原图像是为了在计算误差和传过来的带正负的速度进行计算
        auto rmse = evalRMSE(params_tmp);//计算参数的误差
        cout<<rmse<<endl;
        if (rmse > 2)//超过提前设定的最大误差，则拟合失败,重新在while下进行拟合（修改之后，但是由于输入的数据和残差方式没变，不论拟合次数怎么样，得到的均方差都差不多，最后直接死循环）
            {   

                cout<<summary.BriefReport()<<endl;
                //LOG(INFO)<<"[BUFF_PREDICT]RMSE is too high, Fitting failed!";//LOG打印日志
                //return false;//是不是在这里，输出了之后然后就主函数返回了
            }
        else
            {
                //LOG(INFO)<<"[BUFF_PREDICT]Fitting Succeed! RMSE: "<<rmse;
                params[0] = params_fitting[0] * rotate_sign;//最后放进去再取反，放的还是正的正弦函数
                params[1] = params_fitting[1];
                params[2] = params_fitting[2];
                params[3] = params_fitting[3] * rotate_sign;
                is_params_confirmed = true;
                //cout<<"hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhnew_二次拟合没进行说明这里没有进行"<<endl;
            }
            //cout<<"hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhnew_二次拟合咋没进行"<<endl;
        }
        /*一次拟合完成，准备进行二次拟合，原逻辑语句错误(原来用的是ifelse，但实际上应该用while)，导致一次拟合之后没能进行二次拟合*/
        
           //cout<<"hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhnew_二次拟合了吗"<<endl;//测试中二次拟合和第一次差别不大         
            ceres::Problem problem;
            ceres::Solver::Options options;
            ceres::Solver::Summary summary;       // 优化信息
            double phase;//一个待拟合参数，初相角
            for (auto target_info : history_info)//相当于数据的放入（数据存在于信息队列中）
            {
                problem.AddResidualBlock(     // 向问题中添加误差项
                // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
                    new ceres::AutoDiffCostFunction<CURVE_FITTING_COST_PHASE, 1, 1> ( 
                        new CURVE_FITTING_COST_PHASE ((float)(target_info.timestamp) ,
                                                                    (target_info.speed - params[3]) * rotate_sign, params[0], params[1], params[3])//里面放入5个输入量（时间、速度、a、w、b），其中第二个量有待思考
                    ),
                    new ceres::CauchyLoss(1e1),         //柯西核函数用于抑制数据中的异常值所带来的影响（能够降低异常值所占的权重）
                    &phase                              // 待估计参数
                );
            }
            //设置初相角的上下限并solve
            problem.SetParameterLowerBound(&phase,0,-CV_PI);
            problem.SetParameterUpperBound(&phase,0,CV_PI);
            ceres::Solve(options, &problem, &summary);
            //得到新的拟合4参量，之后与旧的4参量进行误差大小比较
            double params_new[4] = {params[0], params[1], phase, params[3]};
            auto old_rmse = evalRMSE(params);
            auto new_rmse = evalRMSE(params_new);
            if (new_rmse < old_rmse)
            {   
                //LOG(INFO)<<"[BUFF_PREDICT]Params Updated! RMSE: "<<new_rmse;
                params[2] = phase;//更新初相角
            }
            cout<<"hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhnew_rmse"<<endl;//这里也没有输出
            cout<<"old rmse"<<old_rmse<<endl;
            cout<<"RMSE:"<<new_rmse<<endl;
        
    
    for (auto param : params)
        cout<<param<<" ";
        cout<<"hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhparam"<<endl;
#ifdef DRAW_PREDICT
    
        std::vector<double> plt_time;
        std::vector<double> plt_speed;
        std::vector<double> plt_fitted;
        for (auto target_info : history_info)
        {
            auto t = (float)(target_info.timestamp) ;
            plt_time.push_back(t);
            plt_speed.push_back(target_info.speed);//放进去的y值数据
            plt_fitted.push_back(params[0] * sin (params[1] * t + params[2]) + params[3]);//拟合出来的y值数据
        }
        plt::clf();
        plt::plot(plt_time, plt_speed,"bx");
        plt::plot(plt_time, plt_fitted,"r-");//这个在实际画的时候没有画出来是为什么，把时间戳的除的1000去掉就能出来了（因为测试用的是秒为单位，实际是毫秒）
        plt::pause(0.001);

    plt::show();
#endif 
}





















