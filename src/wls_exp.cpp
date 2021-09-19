#include <cstddef>
#include <Rcpp.h>
#include <RcppEigen.h>
#include "glmnet_bits/wls.hpp"

using namespace Rcpp;

// Some dummy function for now
// [[Rcpp::export]]
List wls_exp(
    double alm0,
    double almc,
    double alpha,
    int m,
    int no,
    int ni,
    const Eigen::Map<Eigen::MatrixXd> x,
    Eigen::Map<Eigen::VectorXd> r,
    const Eigen::Map<Eigen::VectorXd> v,
    int intr,
    const Eigen::Map<Eigen::VectorXi> ju,
    const Eigen::Map<Eigen::VectorXd> vp,
    const Eigen::Map<Eigen::MatrixXd> cl,
    int nx,
    double thr,
    int maxit,
    Eigen::Map<Eigen::VectorXd> a,
    double aint,
    Eigen::Map<Eigen::VectorXd> g,
    Eigen::Map<Eigen::VectorXi> ia,
    Eigen::Map<Eigen::VectorXi> iy,
    int iz,
    Eigen::Map<Eigen::VectorXi> mm,
    int nino,
    double rsqc,
    int nlp,
    int jerr
    )
{
    glmnet::wls(
            alm0,
            almc,
            alpha,
            m,
            no,
            ni,
            x,
            r,
            v,
            intr,
            ju,
            vp,
            cl,
            nx,
            thr,
            maxit,
            a,
            aint,
            g,
            ia,
            iy,
            iz,
            mm,
            nino,
            rsqc,
            nlp,
            jerr
            );

    return List::create(
            Named("almc")=almc,
            Named("m")=m,
            Named("no")=no,
            Named("ni")=ni,
            Named("r")=r,
            Named("ju")=ju,
            Named("vp")=vp,
            Named("cl")=cl,
            Named("nx")=nx,
            Named("a")=a,
            Named("aint")=aint,
            Named("g")=g,
            Named("ia")=ia,
            Named("iy")=iy,
            Named("iz")=iz,
            Named("mm")=mm,
            Named("nino")=nino,
            Named("rsqc")=rsqc,
            Named("nlp")=nlp,
            Named("jerr")=jerr);
}
