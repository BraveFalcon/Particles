#include <iostream>
#include <cmath>

template<typename T>
class Vector3 {
public:
    T x, y, z;

    Vector3() = default;

    explicit Vector3(const T &all_) {
        x = all_;
        y = all_;
        z = all_;
    }

    Vector3(const T &x_, const T &y_, const T &z_) {
        x = x_;
        y = y_;
        z = z_;
    }

    friend std::ostream &operator<<(std::ostream &out, const Vector3 &v) {
        out << "Vector3(" << v.x << ", " << v.y << ", " << v.z << ")";
        return out;
    }

    T &operator[](size_t i){
        switch (i) {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            default:
                std::cerr << "Error, Vector3 has only 3 coordinates, not " << i + 1 << "\nBad vector: " << *this
                          << std::endl;
                exit(1);
        }
    }

    void set_values(const T &all_) {
        x = all_;
        y = all_;
        z = all_;
    }

    void set_values(const T &x_, const T &y_, const T &z_) {
        x = x_;
        y = y_;
        z = z_;
    }

    bool is_zero() const {
        return (x == 0.0 && y == 0.0 && z == 0.0);
    }

    bool operator==(const Vector3 &other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    bool operator!=(const Vector3 &other) const {
        return !(*this == other);
    }

    void operator*=(const T &k) {
        x *= k;
        y *= k;
        z *= k;
    }

    Vector3 operator*(const T &k) const {
        return Vector3(x * k, y * k, z * k);
    }

    friend Vector3 operator*(const T &k, const Vector3 &v) {
        return Vector3(v.x * k, v.y * k, v.z * k);
    }

    void operator/=(const T &k) {
        x /= k;
        y /= k;
        z /= k;
    }

    Vector3 operator/(const T &k) const {
        return Vector3(x / k, y / k, z / k);
    }

    void operator+=(const Vector3 &other) {
        x += other.x;
        y += other.y;
        z += other.z;
    }

    Vector3 operator+(const Vector3 &other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    void operator-=(const Vector3 &other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
    }

    Vector3 operator-() const {
        return Vector3(-x, -y, -z);
    }

    Vector3 operator-(const Vector3 &other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    double abs() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    double sqr() const {
        return x * x + y * y + z * z;
    }
};

typedef Vector3<double> Vector3d;