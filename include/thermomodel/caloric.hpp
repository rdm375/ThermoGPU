#pragma once
#include <array>
#include <memory>
#include <limits>
namespace thermomodel {
class CaloricModel {
public:
    virtual ~CaloricModel() = default;

    virtual double minimum_temperature() const noexcept {
        return 0.0;
    }

    virtual double maximum_temperature() const noexcept {
        return std::numeric_limits<double>::infinity();
    }
    virtual double cp(double T) const noexcept = 0;
    virtual double enthalpy_delta(double T, double Tref) const noexcept = 0;
    virtual double entropy_thermal_delta(double T, double Tref) const noexcept = 0;
    virtual double minimum_cp(double Tmin, double Tmax) const noexcept = 0;
};
class ConstantCp final : public CaloricModel {
public:
    explicit ConstantCp(double cp);
    double cp(double) const noexcept override { return cp_; }
    double enthalpy_delta(double T,double ref) const noexcept override { return cp_*(T-ref); }
    double entropy_thermal_delta(double T,double ref) const noexcept override;
    double minimum_cp(double,double) const noexcept override { return cp_; }
private: double cp_;
};
// cp(T) = sum_{k=0}^4 a[k] (T/Tscale)^k, SI units.
class PolynomialCp final : public CaloricModel {
public:
    explicit PolynomialCp(std::array<double,5> a, double Tscale=1000.0);
    double cp(double T) const noexcept override;
    double enthalpy_delta(double T,double ref) const noexcept override;
    double entropy_thermal_delta(double T,double ref) const noexcept override;
    double minimum_cp(double Tmin,double Tmax) const noexcept override;
private:
    std::array<double,5> a_;
    double scale_;
};
} // namespace thermomodel
