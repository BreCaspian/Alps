#include <iostream>

class Vec3{
    public:
    double x, y, z;

    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

    friend std::ostream& operator<<(std::ostream& os, const Vec3& v) {
        os << "(" << v.x << "," << v.y << "," << v.z << ")";
        return os;
    }
};

int main(int argc, char **argv) {
    
    Vec3 a(1.0, 2.0, 3.0);
    Vec3 b(4.0, 5.0, 6.0);

    Vec3 c = a + b;

    std::cout << c << std::endl;

    return 0;
}