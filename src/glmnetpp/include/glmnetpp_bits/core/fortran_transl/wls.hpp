#pragma once
#include <cstddef>
#include <Eigen/Core>
#include <vector>
#include <cmath>

namespace glmnetpp {

enum class wls_state
{
    noop,
    maxit_reached
};

namespace details {

inline wls_state wls_partial_fit(
        int& iz, 
        int& jz,
        int& nlp,
        int& nino,
        int m,
        int maxit,
        int intr,
        double thr,
        double xmz,
        const Eigen::Map<Eigen::VectorXi> ia,
        Eigen::Map<Eigen::VectorXd> r,
        const Eigen::Map<Eigen::MatrixXd> x,
        const Eigen::Map<Eigen::VectorXd> v,
        Eigen::Map<Eigen::VectorXd> a,
        double& aint,
        const std::vector<double>& xv,
        const Eigen::Map<Eigen::VectorXd> vp,
        const Eigen::Map<Eigen::MatrixXd> cl,
        double& rsqc,
        double ab,
        double dem,
        int& jerr)
{
    iz = 1;
    while (1) {
        ++nlp;
        double dlx = 0.0;
        for (int l = 0; l < nino; ++l) {
            auto k = ia(l) - 1;
            auto gk = r.dot(x.col(k));
            auto ak = a(k); 
            auto u = gk + ak * xv[k]; 
            auto au = std::abs(u) - vp(k) * ab;
            if (au < 0.0) { a(k) = 0.0; }
            else {
                a(k) = std::max(cl(0,k),
                        std::min(cl(1,k), 
                            std::copysign(au, u) / (xv[k] + vp(k) * dem)
                            ) );
            }
            if (a(k) == ak) continue;
            auto d = a(k) - ak;
            rsqc += d * (2.0 * gk - d * xv[k]);
            r -= (d * v.array() * x.col(k).array()).matrix(); 
            dlx = std::max(xv[k] * d * d, dlx);
        }

        // updating of intercept term
        double d = 0.0; 
        if (intr != 0) { d = r.sum() / xmz; }
        if (d != 0.0) {
            aint += d;
            rsqc += d * (2.0 * r.sum() - d * xmz);
            dlx = std::max(dlx, xmz * d * d);
            r -= d * v;
        }

        if (dlx < thr) break;
        if (nlp > maxit) { 
            jerr = -m; 
            return wls_state::maxit_reached; 
        }
    }

    // set jz = 0 so that we have to go into :again: tag
    // to check KKT conditions.
    jz = 0;

    return wls_state::noop;
}

} // namespace details

/*
 * Experimental C++ implementation of WLS
 *
 * alm0: previous lambda value
 * almc: current lambda value
 * alpha: alpha
 * m: current lambda iteration no.
 * no: no of observations
 * ni: no of variables
 * x: x matrix
 * r: weighted residual! v * (y - yhat)
 * v: weights
 * intr: 1 if fitting intercept, 0 otherwise
 * ju: ju(k) = 1 if feature k is included for consideration in model
 * vp: relative penalties on features (sum to ni)
 * cl: coordinate limits
 * nx: limit for max no. of variables ever to be nonzero
 * thr: threshold for dlx
 * maxit: max no. of passes over the data for all lambda values
 * a, aint: warm start for coefs, intercept
 * g: abs(dot_product(r,x(:,j)))
 * ia: mapping nino to k
 * iy: ever-active set (for compatibility with sparse version)
 * iz: flag for loop. 0 for first lambda, 1 subsequently
 * mm: mapping k to nino
 * nino: no. of features that have ever been nonzero
 * rsqc: R^2
 * nlp: no. of passes over the data
 * jerr: error code
 */

// TODO:
// - xv doesn't need to be allocated every time

inline void wls(
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
    double& aint,
    Eigen::Map<Eigen::VectorXd> g,
    Eigen::Map<Eigen::VectorXi> ia,
    Eigen::Map<Eigen::VectorXi> iy,
    int& iz,
    Eigen::Map<Eigen::VectorXi> mm,
    int& nino,
    double& rsqc,
    int& nlp,
    int& jerr
        )
{
    std::vector<double> xv(ni);

    // compute g initialization
    for (int j = 0; j < ni; ++j) {
        if (ju(j) == 0) continue; 
        g(j) = std::abs(r.dot(x.col(j)));
    }

    // compute xv
    for (int j = 0; j < ni; ++j) {
        if (iy(j) > 0) {
            xv[j] = v.dot(x.col(j).array().square().matrix());
        }
    }

    // compute xmz (for intercept later)
    double xmz = v.sum();

    // ab: lambda * alpha, dem: lambda * (1 - alpha)
    double ab = almc * alpha; 
    double dem = almc * (1.0 - alpha);

    // strong rules: iy(k) = 1 if we don't discard feature k
    double tlam = alpha * (2.0 * almc - alm0);
    for (int k = 0; k < ni; ++k) {
        if (iy(k) == 1 || 
            ju(k) == 0) continue; 
        if (g(k) > tlam * vp(k)) {
            iy(k) = 1; 
            xv[k] = v.dot(x.col(k).array().square().matrix());
        }
    }
    
    int jz = 1;

    if (iz*jz != 0) {
        auto state = details::wls_partial_fit(
            iz, jz, nlp, nino, m, maxit, intr,
            thr, xmz, ia, r, x, v, a, aint, xv, 
            vp, cl, rsqc, ab, dem ,jerr);
        if (state == wls_state::maxit_reached) return;
    }

    while (1) {

        bool converged = false;
        while (1) {

            // :again: 
            ++nlp; 
            double dlx = 0.0;

            for (int k = 0; k < ni; ++k) {

                // if feature discarded by strong rules, skip over it"
                if (iy(k) == 0) continue;

                // check if ST threshold for descent is met
                // if it goes to 0, set a(k) = 0.0
                // if not, set a(k) to the post-gradient descent value
                // u is the kth partial residual
                auto gk = r.dot(x.col(k));
                auto ak = a(k);
                auto u = gk + ak * xv[k];
                auto au = std::abs(u) - vp(k) * ab;
                if (au < 0.0) { a(k) = 0.0; }
                else {
                    a(k) = std::max(cl(0,k),
                            std::min(cl(1,k), 
                                std::copysign(au,u) / (xv[k] + vp(k) * dem)));
                }

                // if the update didn't change the coefficient value, go to
                // the next feature/variable
                if (a(k) == ak) continue;

                // if coef for feature k was previously 0, we now have a 
                // new non-zero coef. update nino, mm(k) and ia(nino).
                if (mm(k) == 0) {
                    ++nino;
                    if (nino > nx) break;
                    mm(k) = nino; 
                    ia(nino-1) = k+1; // TODO: this is so ugly, but logically correct
                }

                // update residual r, rsqc, and dlx (diff exp from wls)
                auto d = a(k) - ak;
                rsqc += d * (2.0 * gk - d * xv[k]);
                r -= (d * v.array() * x.col(k).array()).matrix();
                dlx = std::max(xv[k] * d * d, dlx);
            }

            // if we've gone over max no. of vars allowed to enter all
            // models, leave the loop
            if (nino > nx) break;

            // updating of intercept term
            double d = 0.0; 
            if (intr != 0) { d = r.sum() / xmz; }
            if (d != 0.0) {
                aint += d;
                rsqc += d * (2.0 * r.sum() - d * xmz);
                dlx = std::max(dlx, xmz * d * d);
                r -= d * v;
            }

            // in wls, this leads to KKT checks. here, we exit
            // the loop instead.
            if (dlx < thr) {
                bool ixx = false;
                for (int k = 0; k < ni; ++k) {
                    if (iy(k) == 1 || ju(k) == 0) continue; 
                    g(k) = std::abs(r.dot(x.col(k)));
                    if (g(k) > ab * vp(k)) {
                       iy(k) = 1; 
                       xv[k] = v.dot(x.col(k).array().square().matrix());
                       ixx = true;
                    }
                }
                if (!ixx) {
                    converged = true;
                    break;
                }
            }

            else break;

        } // end :again: while

        if ((nino > nx) || converged) break;

        // if we've gone over max iterations, return w error
        if (nlp > maxit) { jerr = -m; return; }

        // this is like the :b: loop in wls (M)
        auto state = details::wls_partial_fit(
            iz, jz, nlp, nino, m, maxit, intr,
            thr, xmz, ia, r, x, v, a, aint, xv, 
            vp, cl, rsqc, ab, dem ,jerr);
        if (state == wls_state::maxit_reached) return;

    } // end outer while
}

} // namespace glmnetpp
