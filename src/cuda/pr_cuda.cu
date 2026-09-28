// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/batch.hpp"
#include "thermogpu/peng_robinson.hpp"
#include "thermogpu/cuda_diagnostics.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace thermogpu {
namespace {
constexpr int max_components = 32;
constexpr double sqrt2 = 1.41421356237309504880168872420969808;
constexpr double omega_a = 0.4572355289213822;
constexpr double omega_b = 0.0777960739038884;

struct DeviceComponent { double Tc, Pc, omega, mw; };

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::ostringstream os; os << what << ": " << cudaGetErrorString(e);
        throw std::runtime_error(os.str());
    }
}

template<class T> struct DeviceBuffer {
    T* p{};
    explicit DeviceBuffer(std::size_t n=0) { if (n) cuda_check(cudaMalloc(&p, n*sizeof(T)), "cudaMalloc"); }
    ~DeviceBuffer() { if (p) cudaFree(p); }
    DeviceBuffer(const DeviceBuffer&) = delete; DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__device__ double clampd(double x, double lo, double hi) { return fmin(fmax(x,lo),hi); }
__device__ double residual(double Z,double A,double B) {
    return Z*Z*Z-(1.0-B)*Z*Z+(A-3.0*B*B-2.0*B)*Z-(A*B-B*B-B*B*B);
}

__device__ int real_roots(double A, double B, double roots[3]) {
    const double aa=B-1.0, bb=A-3.0*B*B-2.0*B, cc=-(A*B-B*B-B*B*B);
    const double p=bb-aa*aa/3.0;
    const double q=2.0*aa*aa*aa/27.0-aa*bb/3.0+cc;
    const double disc=q*q/4.0+p*p*p/27.0;
    constexpr double eps=1.0e-14;
    if (disc > eps) {
        const double s=sqrt(disc);
        const double x1=-q/2.0+s, x2=-q/2.0-s;
        const double x=fabs(x1)>=fabs(x2)?x1:x2;
        const double u=cbrt(x);
        // Stable Cardano: recover the smaller term from u*v=-p/3.
        const double v=(u!=0.0)?-p/(3.0*u):cbrt(x==x1?x2:x1);
        roots[0]=u+v-aa/3.0;
        return 1;
    }
    if (fabs(p)<eps && fabs(q)<eps) { roots[0]=-aa/3.0; return 1; }
    const double r=2.0*sqrt(fmax(0.0,-p/3.0));
    const double arg=clampd((3.0*q/(2.0*p))*sqrt(-3.0/p),-1.0,1.0);
    const double th=acos(arg)/3.0;
    constexpr double pi=3.14159265358979323846264338327950288;
    for(int k=0;k<3;++k) roots[k]=r*cos(th-2.0*pi*k/3.0)-aa/3.0;
    return 3;
}

__device__ int largest_root(double A, double B, double& Z) {
    double roots[3];
    const int nr=real_roots(A,B,roots);
    Z=roots[0];
    for(int k=1;k<nr;++k) Z=fmax(Z,roots[k]);
    return nr;
}

__global__ void pr_batch_kernel(const DeviceComponent* c, const double* kij, int nc,
                                const double* P, const double* T, const double* z, std::size_t ns,
                                double* Zout,double* rho,double* cres,double* lnphi,
                                int* rootclass,unsigned char* ambiguous,int* status) {
    const std::size_t s=blockIdx.x*blockDim.x+threadIdx.x;
    if(s>=ns) return;
    double aalpha[max_components], bi[max_components], sumza[max_components];
    double am=0.0,bm=0.0,mw=0.0;
    for(int i=0;i<nc;++i) {
        const double kap=0.37464+1.54226*c[i].omega-0.26992*c[i].omega*c[i].omega;
        const double q=1.0+kap*(1.0-sqrt(T[s]/c[i].Tc));
        const double x=gas_constant*c[i].Tc;
        aalpha[i]=(omega_a*x*x/c[i].Pc)*q*q;
        bi[i]=omega_b*gas_constant*c[i].Tc/c[i].Pc;
        sumza[i]=0.0;
        bm += z[s*nc+i]*bi[i];
        mw += z[s*nc+i]*c[i].mw;
    }
    for(int i=0;i<nc;++i) for(int j=0;j<nc;++j) {
        const double aij=sqrt(aalpha[i]*aalpha[j])*(1.0-kij[i*nc+j]);
        sumza[i] += z[s*nc+j]*aij;
        am += z[s*nc+i]*z[s*nc+j]*aij;
    }
    const double A=am*P[s]/(gas_constant*gas_constant*T[s]*T[s]);
    const double B=bm*P[s]/(gas_constant*T[s]);
    double Z; const int nr=largest_root(A,B,Z);
    if (!(Z>B) || !(am>0.0) || !(bm>0.0) || !(B>0.0)) { status[s]=1; return; }
    const double num=Z+(1.0+sqrt2)*B, den=Z+(1.0-sqrt2)*B;
    if (!(num>0.0) || !(den>0.0)) { status[s]=2; return; }
    const double L=log(num/den);
    Zout[s]=Z;
    cres[s]=residual(Z,A,B);
    rho[s]=P[s]*mw/(Z*gas_constant*T[s]);
    rootclass[s]=(nr==1)?0:1;
    ambiguous[s]=(nr>1)?1:0;
    for(int i=0;i<nc;++i) {
        const double bibm=bi[i]/bm;
        const double attraction=2.0*sumza[i]/am-bibm;
        lnphi[s*nc+i]=bibm*(Z-1.0)-log(Z-B)-A/(2.0*sqrt2*B)*attraction*L;
    }
    status[s]=0;
}
__global__ void diagnose_kernel(const DeviceComponent* c, const double* kij, int nc,
                                double P, double T, const double* z, CudaPrDiagnostic* out) {
    if (blockIdx.x || threadIdx.x) return;
    CudaPrDiagnostic d{}; d.component_count=nc;
    double am=0.0,bm=0.0,mw=0.0;
    for(int i=0;i<nc;++i) {
        d.kappa[i]=0.37464+1.54226*c[i].omega-0.26992*c[i].omega*c[i].omega;
        const double qq=1.0+d.kappa[i]*(1.0-sqrt(T/c[i].Tc));
        d.alpha[i]=qq*qq;
        const double x=gas_constant*c[i].Tc;
        d.a[i]=omega_a*x*x/c[i].Pc; d.a_alpha[i]=d.a[i]*d.alpha[i];
        d.b[i]=omega_b*gas_constant*c[i].Tc/c[i].Pc;
        bm+=z[i]*d.b[i]; mw+=z[i]*c[i].mw;
    }
    for(int i=0;i<nc;++i) for(int j=0;j<nc;++j) {
        const double x=sqrt(d.a_alpha[i]*d.a_alpha[j])*(1.0-kij[i*nc+j]);
        d.aij[i*nc+j]=x; d.sum_z_aij[i]+=z[j]*x; am+=z[i]*z[j]*x;
    }
    d.a_m=am; d.b_m=bm; d.molar_mass=mw;
    d.A=am*P/(gas_constant*gas_constant*T*T); d.B=bm*P/(gas_constant*T);
    d.cubic_a=d.B-1.0; d.cubic_b=d.A-3.0*d.B*d.B-2.0*d.B;
    d.cubic_c=-(d.A*d.B-d.B*d.B-d.B*d.B*d.B);
    d.p=d.cubic_b-d.cubic_a*d.cubic_a/3.0;
    d.q=2.0*d.cubic_a*d.cubic_a*d.cubic_a/27.0-d.cubic_a*d.cubic_b/3.0+d.cubic_c;
    d.discriminant=d.q*d.q/4.0+d.p*d.p*d.p/27.0;
    constexpr double eps=1e-14;
    if(d.discriminant<=eps && !(fabs(d.p)<eps&&fabs(d.q)<eps)) {
        d.trig_arg=clampd((3*d.q/(2*d.p))*sqrt(-3/d.p),-1,1);
        d.theta=acos(d.trig_arg)/3;
    }
    d.root_count=real_roots(d.A,d.B,d.roots);
    d.Z=d.roots[0]; for(int k=1;k<d.root_count;++k)d.Z=fmax(d.Z,d.roots[k]);
    d.cubic_residual=residual(d.Z,d.A,d.B); d.density=P*mw/(d.Z*gas_constant*T);
    const double L=log((d.Z+(1+sqrt2)*d.B)/(d.Z+(1-sqrt2)*d.B));
    for(int i=0;i<nc;++i){double bibm=d.b[i]/bm, attr=2*d.sum_z_aij[i]/am-bibm; d.ln_phi[i]=bibm*(d.Z-1)-log(d.Z-d.B)-d.A/(2*sqrt2*d.B)*attr*L;}
    *out=d;
}

}

struct CudaBatchWorkspace::Impl {
    std::size_t ns{}, nc{};
    DeviceBuffer<DeviceComponent> dc;
    DeviceBuffer<double> dk, dP, dT, dz, dZ, dr, dres, dln;
    DeviceBuffer<int> drc, dst;
    DeviceBuffer<unsigned char> damb;

    Impl(const Mixture& mixture, const MixtureBatch& batch)
        : ns(batch.size()), nc(mixture.size()), dc(nc), dk(nc*nc), dP(ns), dT(ns),
          dz(ns*nc), dZ(ns), dr(ns), dres(ns), dln(ns*nc), drc(ns), dst(ns), damb(ns) {
        Mixture invariant=mixture;
        if(invariant.mole_fractions.size()!=invariant.size()) invariant.mole_fractions.assign(invariant.size(),0.0);
        if(!invariant.mole_fractions.empty()){ invariant.mole_fractions.assign(invariant.size(),0.0); invariant.mole_fractions[0]=1.0; }
        validate_mixture(invariant); validate_batch(mixture,batch);
        if(nc>max_components) throw std::invalid_argument("CUDA V1 baseline supports at most 32 mixture components");
        if(ns==0) return;
        std::vector<DeviceComponent> hc(nc);
        for(std::size_t i=0;i<nc;++i) hc[i]={mixture.components[i].critical_temperature_K,mixture.components[i].critical_pressure_Pa,mixture.components[i].acentric_factor,mixture.components[i].molar_mass_kg_per_mol};
        cuda_check(cudaMemcpy(dc.p,hc.data(),nc*sizeof(DeviceComponent),cudaMemcpyHostToDevice),"copy components H2D");
        cuda_check(cudaMemcpy(dk.p,mixture.binary_interactions.data(),nc*nc*sizeof(double),cudaMemcpyHostToDevice),"copy kij H2D");
        cuda_check(cudaMemcpy(dP.p,batch.pressure_Pa.data(),ns*sizeof(double),cudaMemcpyHostToDevice),"copy P H2D");
        cuda_check(cudaMemcpy(dT.p,batch.temperature_K.data(),ns*sizeof(double),cudaMemcpyHostToDevice),"copy T H2D");
        cuda_check(cudaMemcpy(dz.p,batch.mole_fractions.data(),ns*nc*sizeof(double),cudaMemcpyHostToDevice),"copy z H2D");
    }

    void run() {
        if(ns==0) return;
        constexpr int threads=128; const int blocks=static_cast<int>((ns+threads-1)/threads);
        pr_batch_kernel<<<blocks,threads>>>(dc.p,dk.p,static_cast<int>(nc),dP.p,dT.p,dz.p,ns,dZ.p,dr.p,dres.p,dln.p,drc.p,damb.p,dst.p);
        cuda_check(cudaGetLastError(),"launch PR CUDA kernel");
        cuda_check(cudaDeviceSynchronize(),"execute PR CUDA kernel");
    }

    MixtureBatchResult download() const {
        MixtureBatchResult out; out.state_count=ns; out.component_count=nc;
        out.Z.resize(ns); out.density_kg_per_m3.resize(ns); out.cubic_residual.resize(ns);
        out.ln_phi.resize(ns*nc); out.root_classification.resize(ns); out.root_selection_ambiguous.resize(ns);
        if(ns==0) return out;
        std::vector<int> rc(ns),status(ns);
        cuda_check(cudaMemcpy(out.Z.data(),dZ.p,ns*sizeof(double),cudaMemcpyDeviceToHost),"copy Z D2H");
        cuda_check(cudaMemcpy(out.density_kg_per_m3.data(),dr.p,ns*sizeof(double),cudaMemcpyDeviceToHost),"copy density D2H");
        cuda_check(cudaMemcpy(out.cubic_residual.data(),dres.p,ns*sizeof(double),cudaMemcpyDeviceToHost),"copy residual D2H");
        cuda_check(cudaMemcpy(out.ln_phi.data(),dln.p,ns*nc*sizeof(double),cudaMemcpyDeviceToHost),"copy lnphi D2H");
        cuda_check(cudaMemcpy(rc.data(),drc.p,ns*sizeof(int),cudaMemcpyDeviceToHost),"copy root class D2H");
        cuda_check(cudaMemcpy(out.root_selection_ambiguous.data(),damb.p,ns*sizeof(unsigned char),cudaMemcpyDeviceToHost),"copy ambiguity D2H");
        cuda_check(cudaMemcpy(status.data(),dst.p,ns*sizeof(int),cudaMemcpyDeviceToHost),"copy status D2H");
        for(std::size_t s=0;s<ns;++s){
            if(status[s]) throw std::domain_error("CUDA PR kernel encountered invalid thermodynamic logarithm/domain state at batch index "+std::to_string(s));
            out.root_classification[s]=(rc[s]==0)?RootClassification::one_real:RootClassification::three_real;
        }
        return out;
    }
};

CudaBatchWorkspace::CudaBatchWorkspace(const Mixture& mixture, const MixtureBatch& batch)
    : impl_(std::make_unique<Impl>(mixture,batch)) {}
CudaBatchWorkspace::~CudaBatchWorkspace() = default;
CudaBatchWorkspace::CudaBatchWorkspace(CudaBatchWorkspace&&) noexcept = default;
CudaBatchWorkspace& CudaBatchWorkspace::operator=(CudaBatchWorkspace&&) noexcept = default;
void CudaBatchWorkspace::run() { impl_->run(); }
MixtureBatchResult CudaBatchWorkspace::download() const { return impl_->download(); }

MixtureBatchResult evaluate_mixture_batch_cuda(const Mixture& mixture, const MixtureBatch& batch) {
    CudaBatchWorkspace workspace(mixture,batch);
    workspace.run();
    return workspace.download();
}

CudaPrDiagnostic diagnose_mixture_cuda(const Mixture& mixture, const State& state) {
    validate_mixture(mixture);
    const auto nc=mixture.size(); if(nc>max_components) throw std::invalid_argument("CUDA diagnostics supports at most 32 components");
    std::vector<DeviceComponent> hc(nc); for(std::size_t i=0;i<nc;++i) hc[i]={mixture.components[i].critical_temperature_K,mixture.components[i].critical_pressure_Pa,mixture.components[i].acentric_factor,mixture.components[i].molar_mass_kg_per_mol};
    DeviceBuffer<DeviceComponent> dc(nc); DeviceBuffer<double> dk(nc*nc), dz(nc); DeviceBuffer<CudaPrDiagnostic> dd(1);
    cuda_check(cudaMemcpy(dc.p,hc.data(),nc*sizeof(DeviceComponent),cudaMemcpyHostToDevice),"diag components H2D");
    cuda_check(cudaMemcpy(dk.p,mixture.binary_interactions.data(),nc*nc*sizeof(double),cudaMemcpyHostToDevice),"diag kij H2D");
    cuda_check(cudaMemcpy(dz.p,mixture.mole_fractions.data(),nc*sizeof(double),cudaMemcpyHostToDevice),"diag z H2D");
    diagnose_kernel<<<1,1>>>(dc.p,dk.p,(int)nc,state.pressure_Pa,state.temperature_K,dz.p,dd.p); cuda_check(cudaGetLastError(),"launch diagnostic kernel"); cuda_check(cudaDeviceSynchronize(),"execute diagnostic kernel");
    CudaPrDiagnostic out; cuda_check(cudaMemcpy(&out,dd.p,sizeof(out),cudaMemcpyDeviceToHost),"diagnostic D2H"); return out;
}

} // namespace thermogpu
