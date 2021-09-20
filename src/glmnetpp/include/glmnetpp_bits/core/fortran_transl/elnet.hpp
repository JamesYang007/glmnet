#pragma once 
#include <Eigen/Core>

namespace glmnetpp {

void setpb(int*);

enum class elnet_method
{
    u_method,
    n_method
};

struct StandardBase
{
    template <class XType
            , class YType
            , class WType
            , class JUType
            , class VecType
            , class ValueType>
    static inline void eval(
        XType& x, YType& y, 
        WType& w, int isd, int intr, const JUType& ju,
        VecType& xm, VecType& xs,
        ValueType& ym, ValueType& ys, VecType& xv)
    {
        using vec_t = VecType;

        size_t ni = x.cols();

        w /= w.sum(); 
        vec_t v = w.sqrt();

        // without intercept
        if (intr == 0) { 
            ym = 0.0; 
            y.array() *= v.array();
            ys = y.norm(); 
            y /= ys;

            xm.array() = 0.0;

            for (size_t j = 0; j < ni; ++j) {
                if (ju(j) == 0) continue; 
                auto x_j = x.col(j);
                x_j.array() *= v.array();
                xv(j) = x_j.squaredNorm();
                if (isd != 0) {
                    auto xbq = v.dot(x_j); 
                    xbq *= xbq;
                    auto vc = xv(j) - xbq;
                    xs(j) = std::sqrt(vc); 
                    x_j /= xs(j); 
                    xv(j) = 1.0 + xbq / vc;
                } 
                else { 
                    xs(j) = 1.0; 
                }
            }
        }

        // with intercept
        else {
            for (size_t j = 0; j < ni; ++j) {
                if (ju(j) == 0) continue;
                auto x_j = x.col(j);
                xm(j) = w.dot(x_j); 
                x_j.array() = v * (x_j.array() - xm(j));
                xv(j) = x_j.squaredNorm(); 
                if (isd > 0) {
                    xs(j) = std::sqrt(xv(j));
                }
            }       

            if (isd == 0) { 
                xs.array() = 1.0; 
            }
            else {
                for (size_t j = 0; j < ni; ++j) {
                    if (ju(j) == 0) continue; 
                    x.col(j) /= xs(j);
                }
                xv.array() = 1.0;
            }

            ym = w.dot(y); 
            y.array() = v * (y.array() - ym); 
            ys = y.norm(); 
            y /= ys;
        }
    }
};

template <elnet_method method>
struct Standard;

template <>
struct Standard<elnet_method::u_method> 
    : private StandardBase
{
private:
    using base_t = StandardBase;
    
public:
    template <class XType
            , class YType
            , class WType
            , class JUType
            , class VecType
            , class ValueType>
    static inline void eval(
        XType& x, YType& y, 
        WType& w, int isd, int intr, const JUType& ju,
        VecType& g, VecType& xm, VecType& xs,
        ValueType& ym, ValueType& ys, VecType& xv)
    {
        size_t ni = x.cols();
        base_t::eval(x, y, w, isd, intr, ju, g, xm, xs, ym, ys, xv); 
        g.setZero();
        for (size_t j = 0; j < ni; ++j) {
            if (ju(j) != 0) {
                g(j) = y.dot(x.col(j));
            }
        }
    }
};

template <>
struct Standard<elnet_method::n_method>
    : StandardBase
{
private:
    using base_t = StandardBase;
public:
    using base_t::eval;
};

template <class ValueType>
class ElnetOutput
{
    using value_t = ValueType;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;
    using size_vec_t = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;
    using idx_vec_t = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;

public:

    ElnetOutput(size_t n_lmda = 0,
                size_t max_n_nonzero_coeff = 0)
        : lmu{n_lmda}
        , a0(lmu)
        , ca(max_n_nonzero_coeff, lmu)
        , ia(max_n_nonzero_coeff)
        , nin(lmu)
        , rsq(lmu)
        , alm(lmu)
    {}

    size_t lmu;         // actual number of lamda values (solutions)
    vec_t a0;           // intercept values for each solution
    mat_t ca;           // compressed coefficient values for each solution
    idx_vec_t ia;       // pointers to compressed coefficients
    size_vec_t nin;     // number of compressed coefficients for each solution
    vec_t rsq;          // R^2 values for each solution
    vec_t alm;          // lamda values corresponding to each solution
    size_t nlp = 0;     // actual number of passes over the data for all lamda values
 
    // error flag
    //   jerr = 0 => no error
    //   jerr > 0 => fatal error - no output returned
    //      jerr < 7777 => memory allocation error
    //      jerr = 7777 => all used predictors have zero variance
    //      jerr = 10000 => maxval(vp) <= 0.0
    //   jerr < 0 => non fatal error - partial output:
    //      Solutions for larger lamdas (1:(k-1)) returned.
    //      jerr = -k => convergence for kth lamda value not reached
    //         after maxit (see above) iterations.
    //      jerr = -10000-k => number of non zero coefficients along path
    //         exceeds nx (see above) at kth lamda value.
    int jerr = 0;           

};

struct ElnetInternalParams
{
    using value_t = double;
    static value_t sml;
    static value_t eps;
    static value_t big;
    static size_t mnlam;
    static value_t rsqmax;
    static value_t pmin;
    static value_t exmx;
    static int itrace;
};

ElnetInternalParams::value_t ElnetInternalParams::sml = 1e-5;
ElnetInternalParams::value_t ElnetInternalParams::eps = 1e-6;
ElnetInternalParams::value_t ElnetInternalParams::big = 9.9e35;
size_t ElnetInternalParams::mnlam = 5;
ElnetInternalParams::value_t ElnetInternalParams::rsqmax = 0.999;
ElnetInternalParams::value_t ElnetInternalParams::pmin = 1e-9;
ElnetInternalParams::value_t ElnetInternalParams::exmx = 250.0;
int ElnetInternalParams::itrace = 0;

ElnetInternalParams get_int_parms() { return {}; }

void chg_fract_dev(double arg) { ElnetInternalParams::sml = arg; }
void chg_dev_max(double arg) { ElnetInternalParams::rsqmax = arg; }
void chg_min_flmin(double arg) { ElnetInternalParams::eps = arg; }
void chg_big(double arg) {  ElnetInternalParams::big = arg; }
void chg_min_lambdas(size_t irg) { ElnetInternalParams::mnlam = irg; }
void chg_min_null_prob(double arg) { ElnetInternalParams::pmin = arg; }
void chg_max_exp(double arg) { ElnetInternalParams::exmx = arg; }
void chg_itrace(int irg) { ElnetInternalParams::itrace = irg; }

template <elnet_method method>
struct ElnetFit;

template <>
struct ElnetFit<elnet_method::u_method>
{

    template <class ValueType
            , class JUType
            , class VPType
            , class CLType
            , class VecType
            , class XType
            , class ULamType>
    static inline void eval(
                ValueType beta, size_t ni, const JUType& ju, const VPType& vp,
                const CLType& cl, const VecType& g, size_t no, size_t ne, size_t nx,
                const XType& x, size_t nlam, ValueType flmin, const ULamType& ulam,
                ValueType thr, size_t maxit, const VecType& xv)
    {
        using value_t = ValueType;
        vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
        ivec_t = Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1>;
        mat_t = Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic>;

        mat_t c(ni, nx);

        // sml,eps,big,mnlam,rsqmax,pmin,exmx,itrace
        auto int_parms = get_int_parms();

        vec_t a(ni);
        vec_t mm(ni);
        ivec_t da(ni);

        auto omb = 1.0 - beta;
        // Begin: added by Naras
        value_t alm = 0.0; 
        value_t alf = 1.0;
        // End: added by Naras
        
        if (flmin < 1.0) {
            auto eqs = std::max(int_parms.eps, flmin); 
            alf = std::pow(eqs, 1.0/(nlam-1));
        } 

        a.setZero(); 
        mm.setZero(); 

        value_t rsq = 0.0; 
        size_t nlp = 0;
        size_t nin = 0; 
        bool iz = false; 
        auto mnl = std::min(int_parms.mnlam, nlam);

        output_t out(n_lmda = 0,
                size_t max_n_nonzero_coeff = 0)
        : lmu{n_lmda}
        , a0(lmu)
        , ca(max_n_nonzero_coeff, lmu)
        , ia(max_n_nonzero_coeff)
        , nin(lmu)
        , rsq(lmu)
        , alm(lmu)
    {}

        for (int m = 0; m < nlam; ++m) {
            if (int_parms.itrace != 0) setpb(&m);
            if (flmin > 1.0) alm = ulam(m);
            else if (m > 2) alm *= alf;
            else if (m == 1) alm = int_parms.big;
            else {
                alm = 0.0;
                for (size_t j = 0; j < ni; ++j) {
                    if (ju(j) == 0) continue; 
                    if (vp(j) < 0.0) continue;
                    alm = std::max(alm, abs(g(j)) / vp(j));
                }
                alm *= alf / std::max(beta, 1e-3);
            }

            auto dem = alm * omb; 
            auto ab = alm * beta; 
            auto rsq0 = rsq; 
            bool jz = true;

            while (1) {

                if(iz*jz != 0) go to :b:; 

                ++nlp; 
                value_t dlx = 0.0;

                for (size_t k = 0; k < ni; ++k) {
                    if (ju(k) == 0) continue;
                    auto ak = a(k); 
                    auto u = g(k) + ak * xv(k); 
                    auto v = abs(u) - vp(k) * ab; 
                    a(k)=0.0;
                    if (v > 0.0) {
                        a(k) = std::max(cl(1,k),
                                std::min(cl(2,k),
                                    std::copysign(v,u) / (xv(k) + vp(k) * dem)));
                    }
                    if (a(k) == ak) continue;
                    if (mm(k) == 0) {
                        ++nin; 
                        if (nin > nx) break;
                    } 
                    
                    for (size_t j = 0; j < ni; ++j) {
                        if (ju(j) == 0) continue;
                        if (mm(j) != 0) {
                            c(j,nin) = c(k,mm(j)); 
                            continue;
                        } 
                        if (j != k) { 
                            c(j,nin) = xv(j); 
                            continue; 
                        }
                        c(j,nin) = x.col(j).dot(x.col(k));
                    }
                    
                    mm(k) = nin; 
                    ia(nin-1) = k+1;

                    del=a(k)-ak; rsq=rsq+del*(2.0*g(k)-del*xv(k));
                 dlx=max(xv(k)*del**2,dlx);
                    <j=1,ni; if(ju(j).ne.0) g(j)=g(j)-c(j,mm(k))*del;>
                }

                if (dlx < thr || nin > nx) break; 
                  if nlp.gt.maxit < jerr=-m; return;>
                  :b: iz=1; da(1:nin)=a(ia(1:nin));
                  loop < nlp=nlp+1; dlx=0.0;
                     <l=1,nin; k=ia(l); ak=a(k); u=g(k)+ak*xv(k); v=abs(u)-vp(k)*ab;
                        a(k)=0.0;
                        if(v.gt.0.0)
                           a(k)=max(cl(1,k),min(cl(2,k),sign(v,u)/(xv(k)+vp(k)*dem)));
                        if(a(k).eq.ak) next;
                        del=a(k)-ak; rsq=rsq+del*(2.0*g(k)-del*xv(k));
                        dlx=max(xv(k)*del**2,dlx);
                        <j=1,nin; g(ia(j))=g(ia(j))-c(ia(j),mm(k))*del;>
                     >
                     if(dlx.lt.thr) exit; if nlp.gt.maxit < jerr=-m; return;>
                  >
                  da(1:nin)=a(ia(1:nin))-da(1:nin);
                  <j=1,ni; if(mm(j).ne.0) next;
                     if(ju(j).ne.0) g(j)=g(j)-dot_product(da(1:nin),c(j,1:nin));
                  >
                  jz=0;
           }
           >
           if nin.gt.nx < jerr=-10000-m;  exit;>
           if(nin.gt.0) ao(1:nin,m)=a(ia(1:nin)); kin(m)=nin;
           rsqo(m)=rsq; almo(m)=alm; lmu=m;
           if(m.lt.mnl) next; if(flmin.ge.1.0) next;
           me=0; <j=1,nin; if(ao(j,m).ne.0.0) me=me+1;> if(me.gt.ne) exit;
           if(rsq-rsq0.lt.sml*rsq) exit; if(rsq.gt.rsqmax) exit;
        }
    }
};

// output: lmu, a0, ca, ia, nin, rsq, alm
// ao -> ca
// kin -> nin
// rsqo -> rsq
// almo -> alm

deallocate(a,mm,c,da);
return;
end;


template <elnet_method method
        , class ValueType
        , class XType
        , class YType
        , class WType
        , class JDType
        , class VPType
        , class CLType
        , class ULAMType>
inline ElnetOutput<ValueType> elnet_un(
    ValueType parm, size_t no, size_t ni, XType& x, YType& y,
    const WType& w, const JDType& jd, const VPType& vp, const CLType& cl,
    size_t ne, size_t nx, size_t nlam, ValueType flmin, const ULAMType& ulam,
    ValueType thr, int isd, int intr, size_t maxit)
{
    using value_t = ValueType;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using ivec_t = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;
    using output_t = ElnetOutput<value_t>;

    vec_t xm(ni);
    vec_t xs(ni);
    vec_t xv(ni);
    vec_t vlam(nlam);
    ivec_t ju(ni);
    vec_t g;    // not used if u_version is false

    chkvars(x, ju);

    if (jd(0) > 0) {
        ju(jd.block(1, jd(0))) = 0;
    }

    if (ju.maxCoeff() < 0) { 
        output_t out(0);        
        out.jerr = 7777; 
        return out; 
    }

    double ym = 0, ys = 0;
    if constexpr (method == elnet_method::u_method) {
        Standard<method>::eval(x, y, w, isd, intr, ju, /**/ g /**/ , xm, xs, ym, ys, xv);
    } else {
        Standard<method>::eval(x, y, w, isd, intr, ju, xm, xs, ym, ys, xv);
    }

    cl /= ys; 
    if (isd > 0) {
        for (size_t j = 0; j < ni; ++j) {
            cl.col(j) *= xs(j);
        }
    } 
    
    if (flmin > 1.0) {
        vlam = ulam / ys;
    }

    auto out = []() {
        if constexpr (method == elnet_method::u_method) {
            return ElnetFit<method>::eval(
                    parm, ni, ju, vp, cl, /**/ g /**/, no, ne, nx, 
                    x, nlam, flmin, vlam, thr, maxit, xv);
        } else {
            return ElnetFit<method>::eval(
                    parm, ni, ju, vp, cl, /**/ y /**/, no, ne, nx, 
                    x, nlam, flmin, vlam, thr, maxit, xv);
        }
    }();

    if (out.jerr > 0) return;

    for (size_t k = 0; k < out.lmu; ++k) {
        out.alm(k) *= ys; 
        auto nk = out.nin(k);
        for (size_t l = 0; l < nk; ++l) {
            // Note: ia entries are 1-indexed indices
            out.ca(l,k) *= ys / xs(out.ia(l)-1);
        }
        out.a0(k) = 0.0;
        if (intr != 0) {
            value_t shift = 0.0;
            for (size_t i = 0; i < nk; ++i) {
                shift += out.ca(i,k) * xm(out.ia(i));
            }
            out.a0(k) = ym - shift;
        }
    }

    return out;
}

template <class XType, class JUType>
inline void chkvars(const XType& x, JUType& ju)
{
    for (size_t j = 0; j < x.cols(); ++j) {
        ju(j) = 0; 
        auto t = x(0,j);
        for (size_t i = 1; i < x.rows(); ++i) {
            if (x(i,j) == t) continue; 
            ju(j) = 1;
            break;
        }
    }
}

/*
 * Fits elastic net with squared-error loss with dense matrix.
 *
 * @param   x           predictor data matrix flat file (overwritten)
 * @param   ka          algorithm flag (covariance=1, naive=2)
 * @param   parm        penalty member index (0 <= parm <= 1)
 *                          = 0.0 => ridge
 *                          = 1.0 => lasso
 * @param   no          number of observations
 * @param   ni          number of predictor variables
 * @param   y           response vector (overwritten)
 * @param   w           observation weights (overwritten)
 * @param   jd          jd(jd(1)+1) = predictor variable deletion flag
 *                          jd(1) = 0  => use all variables
 *                          jd(1) != 0 => do not use variables jd(2)...jd(jd(1)+1)
 * @param   vp          relative penalties for each predictor variable
 *                          vp(j) = 0 => jth variable unpenalized
 * @param   cl          interval constraints on coefficient values (overwritten)
 *                          cl(1,j) = lower bound for jth coefficient value (<= 0.0)
 *                          cl(2,j) = upper bound for jth coefficient value (>= 0.0)
 * @param   ne          maximum number of variables allowed to enter largest model
 *                      (stopping criterion)
 * @param   nx          maximum number of variables allowed to enter all models
 *                      along path (memory allocation, nx > ne).
 * @param   nlam        (maximum) number of lamda values
 * @param   flmin       user control of lamda values (>=0)
 *                          flmin < 1.0 => minimum lamda = flmin*(largest lamda value)
 *                          flmin >= 1.0 => use supplied lamda values (see below)
 * @param   ulam        user supplied lamda values (ignored if flmin < 1.0)
 * @param   thr         convergence threshold for each lamda solution.
 *                      Iterations stop when the maximum reduction in the criterion value
 *                      as a result of each parameter update over a single pass
 *                      is less than thr times the null criterion value.
 *                      (suggested value, thr=1.0e-7)
 * @param   isd         predictor variable standarization flag:
 *                          isd = 0 => regression on original predictor variables
 *                          isd = 1 => regression on standardized predictor variables
 *                      Note: output solutions always reference original
 *                            variables locations and scales.
 * @param   intr        intercept flag
 *                          intr = 0/1 => dont/do include intercept in model
 * @param   maxit       maximum allowed number of passes over the data for all lambda
 *                      values (suggested values, maxit = 1e5)
 *
 * @return  ElnetOutput
 *
 */

template <class ValueType
        , class XType
        , class YType
        , class WType
        , class JDType
        , class VPType
        , class CLType
        , class ULAMType>
inline ElnetOutput<ValueType> elnet(
    int ka, ValueType parm, size_t no, size_t ni, XType& x, YType& y,
    const WType& w, const JDType& jd, const VPType& vp, const CLType& cl,
    size_t ne, size_t nx, size_t nlam, ValueType flmin, const ULAMType& ulam,
    ValueType thr, int isd, int intr, size_t maxit)
{
    using value_t = ValueType;
    using vec_t = Eigen::Matrix<value_t, Eigen::Dynamic, 1>;
    using output_t = ElnetOutput<value_t>;
 
    if (vp.maxCoeff() < 0.0) { 
        output_t out(0);
        out.jerr = 10000; 
        return out; 
    }

    vec_t vq = vp.array().max(0);
    vq *= ni / vq.sum();

    if (ka == 1) {
        return elnet_un<elnet_method::u_method>(
                parm, no, ni, x, y, w, jd, vq, cl,
                ne, nx, nlam, flmin, ulam, thr, isd, intr, maxit);
    }

    return elnet_un<elnet_method::n_method>(
                parm, no, ni, x, y, w, jd, vq, cl,
                ne, nx, nlam, flmin, ulam, thr, isd, intr, maxit);
}

//
// sparse predictor matrix:
//
//   x, ix, jx = predictor data matrix in compressed sparse row format
// other inputs:
//
//
//
//
//
// least-squares utility routines:
//
//
// uncompress coefficient vectors for all solutions:
//
// call solns(ni,nx,lmu,ca,ia,nin,b)
//
// input:
//
//    ni,nx = input to elnet
//    lmu,ca,ia,nin = output from elnet
//
// output:
//
//    b(ni,lmu) = all elnet returned solutions in uncompressed format
//
//
// uncompress coefficient vector for particular solution:
//
// call uncomp(ni,ca,ia,nin,a)
//
// input:
//
//    ni = total number of predictor variables
//    ca(nx) = compressed coefficient values for the solution
//    ia(nx) = pointers to compressed coefficients
//    nin = number of compressed coefficients for the solution
//
// output:
//
//    a(ni) =  uncompressed coefficient vector
//             referencing original variables
//
//
// evaluate linear model from compressed coefficients and
// uncompressed predictor matrix:
//
// call modval(a0,ca,ia,nin,n,x,f);
//
// input:
//
//    a0 = intercept
//    ca(nx) = compressed coefficient values for a solution
//    ia(nx) = pointers to compressed coefficients
//    nin = number of compressed coefficients for solution
//    n = number of predictor vectors (observations)
//    x(n,ni) = full (uncompressed) predictor matrix
//
// output:
//    f(n) = model predictions
//
//
// evaluate linear model from compressed coefficients and
// compressed predictor matrix:
//
// call cmodval(a0,ca,ia,nin,x,ix,jx,n,f);
//
// input:
//
//    a0 = intercept
//    ca(nx) = compressed coefficient values for a solution
//    ia(nx) = pointers to compressed coefficients
//    nin = number of compressed coefficients for solution
//    x, ix, jx = predictor matrix in compressed sparse row format
//    n = number of predictor vectors (observations)
//
// output:
//    f(n) = model predictions

} // namespace glmnetpp
