// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/peng_robinson.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace thermogpu;
namespace fs = std::filesystem;

namespace {
struct ErrorStats { double max_abs{}; double max_rel{}; std::string max_abs_name; std::string max_rel_name; };

constexpr int quantity_width = 30;
constexpr int value_width = 24;
constexpr int error_width = 16;

std::map<std::string,double> read_fixture(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open fixture: "+path.string());
    std::map<std::string,double> v; std::string line; std::getline(in,line);
    while (std::getline(in,line)) { const auto c=line.find(','); if(c!=std::string::npos) v.emplace(line.substr(0,c),std::stod(line.substr(c+1))); }
    return v;
}

bool close(double got,double ref,double rtol=2e-13,double atol=2e-15) { return std::abs(got-ref)<=atol+rtol*std::abs(ref); }

int compare(const std::string& name,double got,const std::map<std::string,double>& ref,ErrorStats& stats,bool verbose,
            double rtol=2e-13,double atol=2e-15) {
    const auto it=ref.find(name); if(it==ref.end()){std::cerr<<"FAIL missing reference "<<name<<'\n';return 1;}
    const double abs_err=std::abs(got-it->second);
    const double rel_err = std::abs(it->second) > 1e-12 ? abs_err/std::abs(it->second) : 0.0;
    if(abs_err>stats.max_abs){stats.max_abs=abs_err;stats.max_abs_name=name;}
    if(std::abs(it->second) > 1e-12 && rel_err>stats.max_rel){stats.max_rel=rel_err;stats.max_rel_name=name;}
    if(verbose) {
        std::cout << std::left << std::setw(quantity_width) << name
                  << std::right << std::scientific << std::setprecision(16)
                  << std::setw(value_width) << got << ' '
                  << std::setw(value_width) << it->second << ' '
                  << std::setw(error_width) << abs_err << ' ';
        if (std::abs(it->second) > 1e-12)
            std::cout << std::setw(error_width) << rel_err;
        else
            std::cout << std::setw(error_width) << "n/a";
        std::cout << std::defaultfloat << '\n';
    }
    if(!close(got,it->second,rtol,atol)){std::cerr<<std::setprecision(17)<<"FAIL "<<name<<"\n  C++       "<<got<<"\n  reference "<<it->second<<"\n  delta     "<<(got-it->second)<<'\n';return 1;}
    return 0;
}

int run_case(const std::string& case_name,const Mixture& mix,const State& state,const fs::path& fixture,bool verbose) {
    const auto ref=read_fixture(fixture); const auto d=evaluate_mixture_diagnostics(mix,state); const auto& r=d.result;
    ErrorStats stats; int failures=0;
    if(verbose){
        const int table_width = quantity_width + 3 * value_width + 3;
        std::cout << "\n=== " << case_name << " ===\n"
                  << std::left << std::setw(quantity_width) << "quantity"
                  << std::right << std::setw(value_width) << "C++" << ' '
                  << std::setw(value_width) << "frozen reference" << ' '
                  << std::setw(error_width) << "abs error" << ' '
                  << std::setw(error_width) << "rel error" << '\n'
                  << std::string(table_width, '-') << '\n';
    }
#define CMP(N,G) failures+=compare((N),(G),ref,stats,verbose)
    CMP("R",gas_constant); CMP("T",state.temperature_K); CMP("P",state.pressure_Pa); CMP("a_m",r.a_m); CMP("b_m",r.b_m);
    CMP("molar_mass",r.molar_mass_kg_per_mol); CMP("A",r.A); CMP("B",r.B);
    failures+=compare("root_count",static_cast<double>(d.roots.count),ref,stats,verbose,0,0); CMP("Z",r.Z); CMP("density",r.density_kg_per_m3);
    failures+=compare("cubic_residual",r.cubic_residual,ref,stats,verbose,0,5e-15);
    for(std::size_t i=0;i<mix.size();++i){const std::string n(mix.components[i].name); CMP("alpha."+n,d.alpha[i]); CMP("a."+n,d.a[i]); CMP("a_alpha."+n,d.a_alpha[i]); CMP("b."+n,d.b[i]); CMP("sum_z_aij."+n,d.sum_z_aij[i]); CMP("ln_phi."+n,r.ln_phi[i]); CMP("phi."+n,std::exp(r.ln_phi[i])); CMP("fugacity."+n,mix.mole_fractions[i]*state.pressure_Pa*std::exp(r.ln_phi[i]));}
    for(std::size_t i=0;i<d.roots.count;++i) failures+=compare("root."+std::to_string(i),d.roots.values[i],ref,stats,verbose,5e-13,5e-15);
#undef CMP
    std::cout<<(failures?"FAIL":"PASS")<<" "<<std::left<<std::setw(29)<<case_name<<" max|delta|="<<std::scientific<<stats.max_abs<<" ("<<stats.max_abs_name<<")  max rel="<<stats.max_rel<<" ("<<stats.max_rel_name<<")"<<std::defaultfloat<<'\n';
    return failures;
}

Component methane(){return {"methane",190.56,4.5992e6,0.01142,0.016043};}
Component ethane(){return {"ethane",305.32,4.872e6,0.0995,0.030070};}
Component co2(){return {"carbon_dioxide",304.13,7.3773e6,0.22394,0.0440095};}
Mixture five(){return {{{"methane",190.56,4.5992e6,0.01142,0.016043},{"ethane",305.32,4.872e6,0.0995,0.030070},{"propane",369.83,4.248e6,0.1523,0.044097},{"nitrogen",126.20,3.3958e6,0.0372,0.0280134},{"carbon_dioxide",304.13,7.3773e6,0.22394,0.0440095}},{0.80,0.08,0.04,0.04,0.04},std::vector<double>(25,0.0)};}
}

int main(int argc,char** argv){
    if(argc<2||argc>3){std::cerr<<"usage: test_pr_frozen_oracle FIXTURE_DIR [--verbose]\n";return 2;}
    const fs::path dir=argv[1]; const bool verbose=argc==3&&std::string(argv[2])=="--verbose"; int failures=0;
    auto pure=[&](const std::string& name,const Component& c,const State& s){return run_case(name,Mixture{{c},{1.0},{0.0}},s,dir/(name+".csv"),verbose);};
    failures+=pure("methane_low_pressure",methane(),{1e3,300});
    failures+=pure("methane_moderate_pressure",methane(),{5e6,300});
    failures+=run_case("methane_ethane_binary",Mixture{{methane(),ethane()},{0.8,0.2},{0,0,0,0}},{5e6,300},dir/"methane_ethane_binary.csv",verbose);
    failures+=run_case("binary_nonzero_kij",Mixture{{methane(),co2()},{0.7,0.3},{0,0.03,0.03,0}},{8e6,300},dir/"binary_nonzero_kij.csv",verbose);
    failures+=run_case("five_component",five(),{8e6,320},dir/"five_component.csv",verbose);
    failures+=run_case("high_pressure_five_component",five(),{5e7,400},dir/"high_pressure_five_component.csv",verbose);
    failures+=run_case("binary_three_root",Mixture{{{"nitrogen",126.1,33.94e5,0.04,0.0280134},{"methane",190.6,46.04e5,0.011,0.01604246}},{0.5,0.5},{0,0,0,0}},{1e6,115},dir/"binary_three_root.csv",verbose);
    failures+=pure("methane_near_critical",methane(),{4.5e6,190});
    std::cout<<"\nValidation campaign: "<<(failures?"FAIL":"PASS")<<" (8 cases)\n"; return failures?1:0;
}
