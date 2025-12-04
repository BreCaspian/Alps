# include <iostream>
# include <ctime>


# include <Eigen/Dense> // 稠密矩阵运算
# include <Eigen/Core>

#define MATRIX_SIZE  5000

int main(int argc, char **argv){

    // Matrix 是 Eigen 中的模板类，Matrix<数据类型，行数，列数>，所有向量和矩阵都可以用Matrix来表示

    // MatrixNd  NxN方阵
    // VectorNd  Nx1列向量
    // RowMatrixNd 1xN行向量
    
    Eigen::Matrix<float,2,3> matrix_23;

    Eigen::Vector3d Vd_3d;// Matrix<double,3,1> Vector; 3x1
    Eigen::RowVector3d RVd_3d; // Matrix<double,1,3> Vector; 1x3
    

    Eigen::Matrix<float,3,1> Mf_3d;
    
    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic; // 未知大小使用动态矩阵
    //matrix_dynamic.resize(MATRIX_SIZE,MATRIX_SIZE);
    //Eigen::MatrixXd matrix_dynamic(MATRIX_SIZE,MATRIX_SIZE);

    Eigen::MatrixXf matrix_DynamicFloatMatrix; // 动态大小 float 矩阵
    Eigen::MatrixXd matrix_DynamicDoubleVector; // 动态大小 double 列向量
    Eigen::Matrix3d matrix_33Double; // 3x3 double矩阵
    Eigen::Vector3d vector_31Double; // 三维 double向量

    matrix_23 << 1, 2, 3, 4, 5, 6;

    std::cout << "matrix 2x3 :\n" << matrix_23 << std::endl; 

    std::cout << "matrix 2x3 :\n" << std::endl;

    for (int i = 0; i < 2; i++){
        for(int j = 0; j < 3; j++) std::cout << matrix_23(i,j) << "\t";
        std::cout << "\n" << std::endl;
    }

    Vd_3d << 3, 2, 1;
    Mf_3d << 4, 5, 6;

    Eigen::Matrix<double,3,3> Matrix_Result = Mf_3d.cast<double>() * Vd_3d.transpose();

    for (int i=0;i<3;i++){
        for (int j=0;j<3;j++){
            std::cout << Matrix_Result(i,j)<< "\t" ;
        }
        std::cout << "\n" << std::endl;
    }

    Eigen::Matrix<double, 2, 1> Matrix_Result_2d = matrix_23.cast<double>() * Mf_3d.cast<double>();

    for (int i = 0; i < 2; i++){
        for (int j = 0;j <1 ;j++){
            std::cout << Matrix_Result_2d(i,j) << "\t";
        }
        std::cout << "\n" << std::endl;
    }

    Eigen::Matrix3d Matrix_Random_1;
    Eigen::Vector2d Matrix_Random_2;
    Matrix_Random_1 = Eigen::Matrix3d::Random();
    Matrix_Random_2 = Eigen::Vector2d::Random();

    std::cout << Matrix_Random_1 << "\t" << "\n" << std::endl;
    std::cout << Matrix_Random_2 << "\t" << "\n" << std::endl;

    // 矩阵常规运算
    std::cout << Matrix_Random_1.transpose() << "\t" << "\n" << std::endl; // 转置
    std::cout << Matrix_Random_1.sum() << "\t" << "\n" << std::endl; // 所有元素求和
    std::cout << Matrix_Random_1.trace() << "\t" << "\n" << std::endl; // 迹 (主对角线所有元素的和)
    std::cout << Matrix_Random_1*10 << "\t" << "\n" << std::endl; // 数乘
    std::cout << Matrix_Random_1.inverse() << "\t" << "\n" << std::endl; // 逆矩阵
    std::cout << Matrix_Random_1.determinant() << "\t" << "\n" << std::endl; // 行列式
    
    // EigenSolver 特征值分解器（eigen 特征值）   SelfAdjoint 自伴随 = 实对称矩阵 （对实数矩阵来说就是对称矩阵）
    // 专门用来计算实对称矩阵的特征值和特征向量
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose()*matrix_33);
    std::cout << eigen_solver.eigenvalues() << std::endl;
    std::cout << eigen_solver.eigenvectors() << std::endl;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix_NN = Eigen::MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE); 

    Matrix_NN = Matrix_NN * Matrix_NN.transpose();
    Eigen::Matrix<double, Eigen::Dynamic, 1> v_Nd = Eigen::MatrixXd::Random(MATRIX_SIZE, 1);

    clock_t time_start = clock();
    
    // time : 853794ms    (N^3 + N^3) + N^2
    // Eigen::Matrix<double, Eigen::Dynamic, 1> x = Matrix_NN.inverse() * v_Nd;  // 想法简单，但比较慢  

    // time : 99392.9ms   (1/3 N^3) + N^2
    // Eigen::Matrix<double, Eigen::Dynamic, 1> x = Matrix_NN.ldlt().solve(v_Nd);   // 正定矩阵推荐，快。 前代 -> 直接除 -> 回代

    // time : 518340ms    2 N^3 + N^2
    Eigen::Matrix<double, Eigen::Dynamic, 1> x = Matrix_NN.colPivHouseholderQr().solve(v_Nd);  // 一般情况，不确定矩阵情况推荐。 带列主元的QR分解，多一个列置换矩阵 

    clock_t time_end = clock();

    std::cout << "time : " << 1000 * (time_end - time_start) / (double) CLOCKS_PER_SEC << "ms \n" << std::endl;

    // std::cout << "x = " << x.transpose() << "\n" << std::endl;



    // YAO YUZHUO 2025-12-04 22:07






















    
    







}
