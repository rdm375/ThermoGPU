#include "thermomodel/peng_robinson.hpp"
#include "thermomodel/detail/pr_tp_state_kernel.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
using namespace thermomodel;
using namespace thermomodel::detail;
int main() {
    const auto c=make_pr_coefficients(0.016043,190.56,4599200.0,0.01142);
    constexpr double nasa[5]={5.14987613,-1.36709788e-2,4.91800599e-5,-4.84743026e-8,1.66693956e-11};
    PolynomialCpKernel polynomial{};
    polynomial.scale=1000.0;
    std::array<double,5> a{};
    double power=1.0;
    for(int k=0;k<5;++k){a[k]=nasa[k]*c.R*power;polynomial.a[k]=a[k];power*=1000.0;}
    const PRPolynomialStateParameters parameters{{c,101325.0},polynomial,298.15};
    const auto cp=std::make_shared<PolynomialCp>(a,1000.0);
    PRParameters q{};
    PengRobinson vapor(cp,q);
    q.root_policy=PRRootPolicy::Liquid;
    PengRobinson liquid(cp,q);
    q.root_policy=PRRootPolicy::RejectMultiple;
    PengRobinson strict(cp,q);
    const PengRobinson* models[]={&vapor,&liquid,&strict};
    const PRTPRootPolicy policies[]={PRTPRootPolicy::Vapor,PRTPRootPolicy::Liquid,PRTPRootPolicy::RejectMultiple};
    std::size_t cases=0,both_success=0,status_disagreements=0,compared=0,over_1e12=0;
    double max_scaled=0,max_rho=0,max_Z=0;
    double worst_T=0,worst_p=0;
    int worst_policy=0,worst_field=-1;
    for(int i=0;i<101;++i){
        const double T=100.0+1100.0*i/100.0;
        for(int j=0;j<101;++j){
            const double p=100.0*std::pow(1.0e6,j/100.0);
            for(int k=0;k<3;++k){
                ++cases;
                const auto prod=models[k]->evaluate_Tp(T,p);
                const auto kernel=pr_evaluate_Tp_polynomial(parameters,T,p,policies[k]);
                const bool ok=bool(prod);
                const bool kok=kernel.status==PRTPStatus::Success;
                if(ok!=kok){++status_disagreements;continue;}
                if(!ok)continue;
                ++both_success;
                const double x[]={prod.value.T,prod.value.p,prod.value.rho,prod.value.Z,prod.value.u,prod.value.h,prod.value.s,prod.value.cv,prod.value.cp,prod.value.sound_speed};
                const auto& z=kernel.state;
                const double y[]={z.T,z.p,z.rho,z.Z,z.u,z.h,z.s,z.cv,z.cp,z.sound_speed};
                for(int f=0;f<10;++f){
                    const double e=std::fabs(x[f]-y[f])/std::max({1.0,std::fabs(x[f]),std::fabs(y[f])});
                    ++compared;
                    if(!(e<=1e-12))++over_1e12;
                    if(e>max_scaled){max_scaled=e;worst_T=T;worst_p=p;worst_policy=k;worst_field=f;}
                    if(f==2)max_rho=std::max(max_rho,e);
                    if(f==3)max_Z=std::max(max_Z,e);
                }
            }
        }
    }
    std::cout<<std::setprecision(17)
        <<"G2e.3d production comparison cases: "<<cases<<'\n'
        <<"G2e.3d both successful: "<<both_success<<'\n'
        <<"G2e.3d success-status disagreements: "<<status_disagreements<<'\n'
        <<"G2e.3d field comparisons: "<<compared<<'\n'
        <<"G2e.3d differences over 1e-12: "<<over_1e12<<'\n'
        <<"G2e.3d maximum scaled difference: "<<max_scaled<<'\n'
        <<"G2e.3d maximum density difference: "<<max_rho<<'\n'
        <<"G2e.3d maximum Z difference: "<<max_Z<<'\n'
        <<"G2e.3d worst case T,p,policy,field: "<<worst_T<<' '<<worst_p<<' '<<worst_policy<<' '<<worst_field<<'\n';
    // This is a characterization test. Differences are measured, not yet rejected.
    return cases==30603 && compared>0 ? 0:1;
}
