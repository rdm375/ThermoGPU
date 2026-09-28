#include "thermogpu/peng_robinson.hpp"
#include "thermogpu/cuda_diagnostics.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
using namespace thermogpu;
namespace {
Mixture mix(double x){return {{{"methane",190.56,4.5992e6,.01142,.016043},{"ethane",305.32,4.872e6,.09950,.030070}},{x,1-x},{0,0,0,0}};}
std::uint64_t ord(double x){auto u=std::bit_cast<std::uint64_t>(x);return(u>>63)?~u:(u|(1ULL<<63));}
std::uint64_t ulp(double a,double b){auto x=ord(a),y=ord(b);return x>y?x-y:y-x;}

struct CubicProbe { double p,q,disc,z; };
CubicProbe solve_probe(double a,double b,double cc){
    constexpr double pi=3.141592653589793238462643383279502884;
    constexpr double eps=1e-14;
    CubicProbe r{};
    r.p=b-a*a/3.0;
    r.q=2.0*a*a*a/27.0-a*b/3.0+cc;
    r.disc=r.q*r.q/4.0+r.p*r.p*r.p/27.0;
    if(r.disc>eps){double sd=std::sqrt(r.disc);r.z=std::cbrt(-r.q/2.0+sd)+std::cbrt(-r.q/2.0-sd)-a/3.0;}
    else if(std::abs(r.p)<eps&&std::abs(r.q)<eps){r.z=-a/3.0;}
    else {double rr=2.0*std::sqrt(std::max(0.0,-r.p/3.0));double arg=(3.0*r.q/(2.0*r.p))*std::sqrt(-3.0/r.p);arg=std::clamp(arg,-1.0,1.0);double th=std::acos(arg)/3.0;r.z=-std::numeric_limits<double>::infinity();for(int k=0;k<3;++k)r.z=std::max(r.z,rr*std::cos(th-2.0*pi*k/3.0)-a/3.0);}
    return r;
}
double poly(double z,double a,double b,double c){return ((z+a)*z+b)*z+c;}
double dpoly(double z,double a,double b){return (3.0*z+2.0*a)*z+b;}
struct RootProbe { double minus_q2, sqrt_disc, arg_plus, arg_minus, u, v, z, residual, deriv, dz_db, z_newton, residual_newton; };
RootProbe root_probe(double a,double b,double c,bool stable){
    RootProbe r{};
    const double p=b-a*a/3.0;
    const double q=2.0*a*a*a/27.0-a*b/3.0+c;
    const double disc=q*q/4.0+p*p*p/27.0;
    r.minus_q2=-q/2.0;
    r.sqrt_disc=std::sqrt(disc);
    r.arg_plus=r.minus_q2+r.sqrt_disc;
    r.arg_minus=r.minus_q2-r.sqrt_disc;
    // Choose the larger-magnitude Cardano argument for u to avoid the
    // cancellation-prone cube root; recover v from u*v=-p/3 when requested.
    const double arg_u=std::abs(r.arg_plus)>=std::abs(r.arg_minus)?r.arg_plus:r.arg_minus;
    const double arg_v=arg_u==r.arg_plus?r.arg_minus:r.arg_plus;
    r.u=std::cbrt(arg_u);
    r.v=stable ? (-p/(3.0*r.u)) : std::cbrt(arg_v);
    r.z=r.u+r.v-a/3.0;
    r.residual=poly(r.z,a,b,c);
    r.deriv=dpoly(r.z,a,b);
    r.dz_db=-r.z/r.deriv;
    r.z_newton=r.z-r.residual/r.deriv;
    r.residual_newton=poly(r.z_newton,a,b,c);
    return r;
}
void probe_row(const char* n,double b,double a,double cc,double reference){auto r=solve_probe(a,b,cc);std::cout<<std::left<<std::setw(24)<<n<<std::right<<std::setw(29)<<b<<std::setw(29)<<r.p<<std::setw(29)<<r.z<<std::setw(29)<<std::abs(r.z-reference)<<std::setw(22)<<ulp(r.z,reference)<<'\n';}
void row(const char*n,double c,double g){double ad=std::abs(g-c),rd=ad/std::max({std::abs(c),std::abs(g),std::numeric_limits<double>::min()});std::cout<<std::left<<std::setw(24)<<n<<std::right<<std::setw(29)<<c<<std::setw(29)<<g<<std::setw(29)<<ad<<std::setw(29)<<rd<<std::setw(22)<<ulp(c,g)<<'\n';}
}
int main(int argc,char**argv){double P=7.68069763952790573e6,T=2.96542617046818691e2,x=4.35330595297648837e-1;if(argc==4){P=std::stod(argv[1]);T=std::stod(argv[2]);x=std::stod(argv[3]);}else if(argc!=1){std::cerr<<"usage: thermogpu_diagnose_state [P_Pa T_K z_CH4]\n";return 2;}auto m=mix(x);State s{P,T};auto c=evaluate_mixture_diagnostics(m,s);auto g=diagnose_mixture_cuda(m,s);
std::cout<<std::scientific<<std::setprecision(17)<<"ThermoGPU CPU/CUDA state diagnostic\nP="<<P<<" T="<<T<<" z=["<<x<<","<<1-x<<"]\n\n"<<std::left<<std::setw(24)<<"quantity"<<std::right<<std::setw(29)<<"CPU"<<std::setw(29)<<"CUDA"<<std::setw(29)<<"abs diff"<<std::setw(29)<<"rel diff"<<std::setw(22)<<"ULP"<<'\n';
for(size_t i=0;i<2;++i){row(("alpha["+std::to_string(i)+"]").c_str(),c.alpha[i],g.alpha[i]);row(("a["+std::to_string(i)+"]").c_str(),c.a[i],g.a[i]);row(("a_alpha["+std::to_string(i)+"]").c_str(),c.a_alpha[i],g.a_alpha[i]);row(("b["+std::to_string(i)+"]").c_str(),c.b[i],g.b[i]);row(("sum_z_aij["+std::to_string(i)+"]").c_str(),c.sum_z_aij[i],g.sum_z_aij[i]);}
for(size_t i=0;i<2;++i)for(size_t j=0;j<2;++j){double ca=std::sqrt(c.a_alpha[i]*c.a_alpha[j])*(1-m.kij(i,j));row(("aij["+std::to_string(i)+","+std::to_string(j)+"]").c_str(),ca,g.aij[i*2+j]);}
row("a_m",c.result.a_m,g.a_m);row("b_m",c.result.b_m,g.b_m);row("A",c.result.A,g.A);row("B",c.result.B,g.B);
double aa=c.result.B-1,bb=c.result.A-3*c.result.B*c.result.B-2*c.result.B,cc=-(c.result.A*c.result.B-c.result.B*c.result.B-c.result.B*c.result.B*c.result.B),p=bb-aa*aa/3,q=2*aa*aa*aa/27-aa*bb/3+cc,disc=q*q/4+p*p*p/27;
row("cubic a",aa,g.cubic_a);row("cubic b",bb,g.cubic_b);row("cubic c",cc,g.cubic_c);row("depressed p",p,g.p);row("depressed q",q,g.q);row("discriminant",disc,g.discriminant);

const double B2 = c.result.B * c.result.B;
const double three_B2 = 3.0 * B2;
const double two_B = 2.0 * c.result.B;
const double A_minus_3B2 = c.result.A - three_B2;
const double b_staged = A_minus_3B2 - two_B;
const double b_fma = std::fma(-3.0 * c.result.B, c.result.B, c.result.A) - two_B;
const double b_fma_all = std::fma(-2.0, c.result.B,
                                  std::fma(-3.0 * c.result.B, c.result.B, c.result.A));

std::cout<<"\nCubic-b arithmetic probe (CUDA reference = production cubic b)\n";
row("B*B", B2, B2);
row("3*(B*B)", three_B2, three_B2);
row("2*B", two_B, two_B);
row("A-3*(B*B)", A_minus_3B2, A_minus_3B2);
row("normal expression", bb, g.cubic_b);
row("explicit staged", b_staged, g.cubic_b);
row("FMA first term", b_fma, g.cubic_b);
row("nested FMA", b_fma_all, g.cubic_b);
const double b_down=std::nextafter(bb,-std::numeric_limits<double>::infinity());const double b_up=std::nextafter(bb,std::numeric_limits<double>::infinity());
std::cout<<"\nCPU cubic sensitivity to coefficient b\n"<<std::left<<std::setw(24)<<"case"<<std::right<<std::setw(29)<<"cubic b"<<std::setw(29)<<"depressed p"<<std::setw(29)<<"CPU analytic Z"<<std::setw(29)<<"abs vs CUDA Z"<<std::setw(22)<<"ULP vs CUDA"<<'\n';
probe_row("CPU b - 1 ULP",b_down,aa,cc,g.Z);probe_row("CPU b",bb,aa,cc,g.Z);probe_row("CPU b + 1 ULP",b_up,aa,cc,g.Z);probe_row("CUDA b",g.cubic_b,aa,cc,g.Z);

const auto rc=root_probe(aa,bb,cc,false);
const auto rs=root_probe(aa,bb,cc,true);
const auto gc=root_probe(g.cubic_a,g.cubic_b,g.cubic_c,false);
const auto gs=root_probe(g.cubic_a,g.cubic_b,g.cubic_c,true);
std::cout<<"\nOne-real-root Cardano conditioning probe\n";
std::cout<<"CPU coefficient set\n";
row("-q/2",rc.minus_q2,rs.minus_q2);row("sqrt(discriminant)",rc.sqrt_disc,rs.sqrt_disc);
row("arg +",rc.arg_plus,rs.arg_plus);row("arg -",rc.arg_minus,rs.arg_minus);
row("u",rc.u,rs.u);row("v",rc.v,rs.v);row("Cardano Z",rc.z,rs.z);
row("residual",rc.residual,rs.residual);row("f'(Z)",rc.deriv,rs.deriv);row("-Z/f'(Z)",rc.dz_db,rs.dz_db);
row("Newton Z",rc.z_newton,rs.z_newton);row("Newton residual",rc.residual_newton,rs.residual_newton);
std::cout<<"\nCUDA coefficient set evaluated on CPU\n";
row("-q/2",gc.minus_q2,gs.minus_q2);row("sqrt(discriminant)",gc.sqrt_disc,gs.sqrt_disc);
row("arg +",gc.arg_plus,gs.arg_plus);row("arg -",gc.arg_minus,gs.arg_minus);
row("u",gc.u,gs.u);row("v",gc.v,gs.v);row("Cardano Z",gc.z,gs.z);
row("residual",gc.residual,gs.residual);row("f'(Z)",gc.deriv,gs.deriv);row("-Z/f'(Z)",gc.dz_db,gs.dz_db);
row("Newton Z",gc.z_newton,gs.z_newton);row("Newton residual",gc.residual_newton,gs.residual_newton);
const double observed_sensitivity=(gc.z-rc.z)/(g.cubic_b-bb);
std::cout<<"\nObserved dZ/db from 1-ULP coefficient perturbation = "<<observed_sensitivity<<"\n";
std::cout<<"Local implicit derivative at CPU root             = "<<rc.dz_db<<"\n";
std::cout<<"ratio observed/predicted                          = "<<observed_sensitivity/rc.dz_db<<"\n";

std::cout<<"\nCPU root_count="<<c.roots.count<<" CUDA root_count="<<g.root_count<<'\n';const std::size_t root_count = std::max(c.roots.count, static_cast<std::size_t>(g.root_count));for(std::size_t k=0;k<root_count;++k)row(("root["+std::to_string(k)+"]").c_str(),c.roots.values[k],g.roots[k]);row("selected Z",c.result.Z,g.Z);row("density",c.result.density_kg_per_m3,g.density);row("residual",c.result.cubic_residual,g.cubic_residual);for(size_t i=0;i<2;++i)row(("ln_phi["+std::to_string(i)+"]").c_str(),c.result.ln_phi[i],g.ln_phi[i]);return 0;}
