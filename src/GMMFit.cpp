//
// Created by lab on 2021/12/8.
//

#include "GMMFit.h"

bool
GMMFit::fit_1d(const VectorXd &in_vec,
               const int K,
               vector<double> &mu,
               vector<double> &sigma,
               bool show_debug_info = false)
{
    assert(K > 0);
    assert(K == mu.size() && K == sigma.size());

    if (show_debug_info)
    {
        cout << TermColor::GREEN() << "====GMMFit::fit_1d====\n" << TermColor::RESET();
        cout << "Initial Guess:\n";
        for (int k = 0; k < K; k++)
        {
            cout << "#" << k << "\tmu=" << mu[k] << "\tsigma=" << sigma[k] << endl;
        }
    }


    // Priors (len will be K), 1 prior per class
    VectorXd priors = VectorXd::Zero(K);
    for (int k = 0; k < K; k++)
        priors(k) = 1.0 / double(K);
    if (show_debug_info)
    {
        cout << "initial priors: " << priors.transpose() << endl;
    }

    for (int itr = 0; itr < 10; itr++)
    {
        if (show_debug_info)
        {
            cout << "---itr=" << itr << endl;
            // Compute Likelihoods ie. P( x_i / b_k ) \forall i=1 to n , \forall k=1 to K
            cout << TermColor::YELLOW() << "Likelihood Computation" << TermColor::RESET() << endl;
        }
        MatrixXd L = MatrixXd::Zero(in_vec.rows(), K);
        for (int k = 0; k < K; k++)
        {
            VectorXd _tmp = GaussianFunction::eval(in_vec, mu[k], sigma[k]);
            L.col(k) = _tmp;
        }
        // cout << "Likelihood for each datapoint (rows); for each class (cols):\n" << L << endl;



        // Posterior
        if (show_debug_info)
            cout << TermColor::YELLOW() << "Posterior Computation (Bayes Rule)" << TermColor::RESET() << endl;
        MatrixXd P = MatrixXd::Zero(in_vec.rows(), K);
        for (int i = 0; i < in_vec.rows(); i++) //loop over datapoints
        {
            // cout << "L.row("<< i << " ) " << L.row(i) * priors << endl;
            double p_x = L.row(i) * priors; // total probability (denominator in Bayes rule)

            P.row(i) = (L.row(i).transpose().array() * priors.array()).matrix() / p_x;
        }
        // cout << "Posterior:\n" << P << endl;



        // Update Mu, Sigma
        if (show_debug_info)
            cout << TermColor::YELLOW() << "Update mu, sigma" << TermColor::RESET() << endl;
        for (int k = 0; k < K; k++)
        {
            double denom = P.col(k).sum();

            // Updated mu
            double mu_new_numerator = P.col(k).transpose() * in_vec;
            double mu_new = mu_new_numerator / denom;

            // updated sigma
            VectorXd _tmp = ((in_vec.array() - mu_new) * (in_vec.array() - mu_new)).matrix();
            double sigma_new_numerator = P.col(k).transpose() * _tmp;

            double sigma_new = sqrt(sigma_new_numerator / denom);
            if (show_debug_info)
            {
                cout << "mu_new_" << k << " " << mu_new << "\t";
                cout << "sigma_new_" << k << " " << sigma_new << endl;
            }
            mu[k] = mu_new;
            sigma[k] = sigma_new;
        }


        // update priors
        if (show_debug_info) cout << TermColor::YELLOW() << "Update priors" << TermColor::RESET() << endl;
        for (int k = 0; k < K; k++)
        {
            priors(k) = (P.col(k).sum() / double(in_vec.size()));
        }
        if (show_debug_info) cout << "updated priors: " << priors.transpose() << endl;

    }


    if (show_debug_info) cout << TermColor::GREEN() << "==== DONE GMMFit::fit_1d====\n" << TermColor::RESET();
}

bool
GMMFit::fit_multivariate(const MatrixXd &in_vec, const int K,
                         vector<VectorXd> &mu, vector<MatrixXd> &sigma, VectorXd &priors)
{
    //
    // Are the Inputs sane?
    assert(K > 0);
    assert(mu.size() == K && sigma.size() == K);
    int D = mu[0].rows(); //data dimension
    assert(D > 0);
    for (int i = 0; i < K; i++)
    {
        assert(mu[i].rows() == D);
        assert(sigma[i].rows() == D && sigma[i].cols() == D);
        assert(GaussianFunction::isValidCovarianceMatrix(sigma[i]));
        if (GaussianFunction::isValidCovarianceMatrix(sigma[i]) == false)
        {
            cout << TermColor::RED() << "[GMMFit::fit_multivariate] Invalid covariance matrix as input\n";
            return false;
        }
    }
    int N = in_vec.cols();
    assert(N > 0 && in_vec.rows() == D);
    cout << TermColor::GREEN() << "====GMMFit::fit_1d====\n" << TermColor::RESET();

    cout << TermColor::iGREEN() << "====\n";
    cout << "Dimensionality of Data: " << D << endl;
    cout << "Number of Input Points: " << N << endl;
    cout << "Requested Number of Gaussians to Fit: " << K << endl;
    cout << "====\n" << TermColor::RESET() << endl;

    //
    // Printout the initial guesses
    cout << "Initial Guess:\n";
    for (int i = 0; i < K; i++)
    {
        cout << TermColor::GREEN() << "#" << i << "\tmu=" << mu[i].transpose() << TermColor::RESET() << endl;
        cout << "sigma=\n" << sigma[i] << endl;
    }


    //
    // ===== Processing Starts =====
    //
    // init Priors for each class
    // VectorXd priors = VectorXd::Zero( K );
    priors = VectorXd::Zero(K);
    for (int k = 0; k < K; k++)
        priors(k) = 1.0 / double(K);
    cout << "initial priors: " << priors.transpose() << endl;

    for (int itr = 0; itr < 10; itr++)
    {
        cout << TermColor::iYELLOW() << "---itr=" << itr << TermColor::RESET() << endl;


        // Compute Likelihood ie. P( x_i/ b_k ) \forall i, \forall k
        cout << TermColor::YELLOW() << "Likelihood Computation" << TermColor::RESET() << endl;
        MatrixXd L = MatrixXd::Zero(N, K);
        for (int k = 0; k < K; k++)
        {
            VectorXd _tmp = GaussianFunction::eval(in_vec, mu[k], sigma[k]);
            L.col(k) = _tmp;
        }
        // cout << "Likelihood for each datapoint (rows); for each class (cols):\n" << L << endl;


        // Posterior
        cout << TermColor::YELLOW() << "Posterior Computation (Bayes Rule)" << TermColor::RESET() << endl;
        MatrixXd P = MatrixXd::Zero(N, K);
        for (int i = 0; i < N; i++) //loop over datapoints
        {
            // cout << "L.row("<< i << " ) " << L.row(i) * priors << endl;
            double p_x = L.row(i) * priors; // total probability (denominator in Bayes rule)

            P.row(i) = (L.row(i).transpose().array() * priors.array()).matrix() / p_x;
        }
        // cout << "Posterior:\n" << P << endl;



        // update mu, sigmas for each class
        //      weighted sum of datapoints by the Posterior.
        cout << TermColor::YELLOW() << "Update mu, sigma" << TermColor::RESET() << endl;
        for (int k = 0; k < K; k++)
        {

            double denom = P.col(k).sum();

            // updated mu
            VectorXd mu_new_numerator = (in_vec * P.col(k));
            VectorXd mu_new = mu_new_numerator / denom;


            // updated sigma
            // cout << " ( in_vec.colwise() - mu_new ) =\n" << ( in_vec.colwise() - mu_new ) << endl;
            // cout << "P.col(k) =\n" << P.col(k) << endl;
            MatrixXd _tmp_w = (in_vec.colwise() - mu_new).array().rowwise() * P.col(k).transpose().array();
            // cout << "_tmp_w = \n" << _tmp_w << endl;
            MatrixXd _tmp = _tmp_w * (in_vec.colwise() - mu_new).transpose();

            MatrixXd sigma_new = _tmp / denom;

            cout << "--\n";
            cout << "mu_new_" << k << " " << mu_new.transpose() << "\n";
            cout << "sigma_new_" << k << "\n" << sigma_new << endl;

            mu[k] = mu_new;
            sigma[k] = sigma_new;
        }



        // update priors
        cout << TermColor::YELLOW() << "Update priors" << TermColor::RESET() << endl;
        for (int k = 0; k < K; k++)
        {
            priors(k) = (P.col(k).sum() / double(N));
        }
        cout << "updated priors: " << priors.transpose() << endl;

    }

    cout << TermColor::GREEN() << "==== Done GMMFit::fit_1d====\n" << TermColor::RESET();

}
