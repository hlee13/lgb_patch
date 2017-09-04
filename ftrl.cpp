#include <unistd.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>

#include <omp.h>

typedef double weight_t;
typedef weight_t z_t;
typedef weight_t n_t;
typedef weight_t w_t;
typedef double feature_val_t;
typedef feature_val_t feat_val_t;
typedef int feature_idx_t;
typedef feature_idx_t feat_idx_t;

typedef std::vector<std::pair<feat_idx_t, feat_val_t> > feature_vec_t;

#define sign(x) (x > 0 ? 1: (x < 0 ? -1: 0))

double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

class FTRLProximal {
public:
    FTRLProximal(int max_feature_cnt = 200000, double alpha = 1, double beta = 0.5, 
            double l1 = 0.01, double l2 = 0.1, double lamd = 1) : _max_feature_cnt(max_feature_cnt),
        _alpha(alpha), _beta(beta), _l1(l1), _l2(l2), _lamd(lamd), _bias(0), _nbias(0) {
            _z.resize(max_feature_cnt, 0);
            _n.resize(max_feature_cnt, 0);
            _w.resize(max_feature_cnt, 0);
    }

    double predict(const feature_vec_t& feature_vec) {
        double sigma = 0;
        for (auto it = feature_vec.begin(); it != feature_vec.end(); it++) {
            feat_idx_t idx = it->first;
            feat_val_t val = it->second;
            sigma += _w[idx] * val;
        }
        return sigma;
    }

    double fit(const feature_vec_t& feature_vec, double y) {
        for (auto it = feature_vec.begin(); it != feature_vec.end(); it++) {
            feat_idx_t idx = it->first;

            z_t& zi = _z[idx];
            if (std::fabs(zi) > _l1) {
                _w[idx] = - (zi - _l1 * sign(zi)) / (_l2 + (_beta + std::sqrt(_n[idx])) / _alpha);
            } else {
            }
        }

        double predict_val = predict(feature_vec);
        double error = predict_val - y;

        for (auto it = feature_vec.begin(); it != feature_vec.end(); it++) {
            feat_idx_t idx = it->first;
            feat_val_t val = it->second;

            double gi = error * val;
            double gg = gi * gi;
            n_t ni = _n[idx];
            double si = (std::sqrt(ni + gg) - std::sqrt(ni)) / _alpha;
            _z[idx] = _z[idx] + gi - si * _w[idx];
            _n[idx] = _n[idx] + gg;
        }

        return error;
    }

    double predict(const std::vector<feat_idx_t>& feat_idxs, const std::vector<feat_val_t>& feat_vals) {
        int input_feature_cnt = static_cast<int>(feat_idxs.size());
        double sigma = _bias;
// #pragma omp parallel for schedule(static, 512)
        for (int i = 0; i < input_feature_cnt; i++) {
            feat_idx_t idx = feat_idxs[i];
            feat_val_t val = feat_vals[i];
            sigma += _w[idx] * val;
        }
        if (std::isnan(sigma)) {
                sigma = _bias;
            printf("bias=[%f]\n", _bias);
            for (int i = 0; i < input_feature_cnt; i++) {
                feat_idx_t idx = feat_idxs[i];
                feat_val_t val = feat_vals[i];
                printf("idx=[%d] val=[%f]\n", idx, val);
                sigma += _w[idx] * val;
            }
            printf("sigma=[%f]\n", sigma);
            exit(1);
        }
        return sigma;
    }

    double fit(const std::vector<feat_idx_t>& feat_idxs, const std::vector<feat_val_t>& feat_vals, double y) {
        int input_feature_cnt = static_cast<int>(feat_idxs.size());
// #pragma omp parallel for schedule(static, 512)
        for (int i = 0; i < input_feature_cnt; i++) {
            feat_idx_t idx = feat_idxs[i];

            z_t& zi = _z[idx];
            if (std::fabs(zi) > _l1) {
                _w[idx] = - (zi - _l1 * sign(zi)) / (_l2 + (_beta + std::sqrt(_n[idx])) / _alpha);
            } else {
            }
        }

        double predict_val = predict(feat_idxs, feat_vals);
        double error = predict_val - y;
        // double error = sign(predict_val - y) / y;

// #pragma omp parallel for schedule(static, 512)
        for (int i = 0; i < input_feature_cnt; i++) {
            feat_idx_t idx = feat_idxs[i];
            feat_val_t val = feat_vals[i];

            double gi = error * val;
            double gg = gi * gi;
            n_t ni = _n[idx];
            double si = (std::sqrt(ni + gg) - std::sqrt(ni)) / _alpha;
            _z[idx] = _z[idx] + gi - si * _w[idx];
            _n[idx] = _n[idx] + gg;
        }

        _bias -= _alpha * error / (_beta + std::sqrt(_nbias));
        _nbias += error * error;
        // _nbias += 1 / (y*y);

        return predict_val;
    }

    void dump_model(const char* file_name = NULL, bool dump_file = false) {
        size_t feature_cnt = _max_feature_cnt;
        if (dump_file) {
            std::ofstream out(file_name); 
            if (not out.is_open()) {
                fprintf(stderr, "open [%s] dump_model error", file_name);
                return;
            }

            out << _bias << std::endl;
            for (size_t i = 0; i < feature_cnt; i++) {
                // if (_w[i] > 1e-10 or _w[i] < -1e-10) {
                //     out << i << ":" << _w[i] << std::endl;
                // }
                out << i << ":" << _w[i] << std::endl;
            }
            out.close();
            return;
        }

        printf("bias=[%f]", _bias);
        for (size_t i = 0; i < feature_cnt; i++) {
            printf(" %lu:%f", i, _w[i]);
        }
        printf("\n");
    }

    void load_model(const char* file_name) {
        const int buffer_size = 1024;
        char line[buffer_size];

        std::ifstream in(file_name);
        if (not in.is_open()) {
            fprintf(stderr, "load model [%s] error\n", file_name);
            return;
        }

        in.getline(line, buffer_size);
        sscanf(line, "%lf", &_bias);
        while (not in.eof()) {
            in.getline(line, buffer_size);
            int    feat_idx;
            double feat_val;
            sscanf(line, "%d:%lf", &feat_idx, &feat_val);

            _w[feat_idx] = feat_val;
        }

        in.close();
    }
private:
    int _max_feature_cnt;
    double _alpha;
    double _beta;
    double _l1;
    double _l2;
    double _lamd;

    std::vector<z_t> _z;
    std::vector<n_t> _n;
    std::vector<w_t> _w;

    double _bias;
    double _nbias;
};

double rand_uniform() {
    return rand() * 1.0 / (RAND_MAX * 1.0);
}

double mock_func(int feature_cnt, feature_vec_t& feature_vec) {
    double sigma = 15;
    int w[] = {2,3,7,9,-1,6, 12, 31, 8, 0};

    // bias
    feature_vec.push_back(std::make_pair(0, 1));
    for (int i = 0; i < feature_cnt; i++) {
        double v = rand_uniform();
        sigma += v * w[i];

        feature_vec.push_back(std::make_pair(i+1, v));
    }
    return sigma;
}

double mock_func(int feature_cnt, std::vector<feat_idx_t>& feat_idxs, std::vector<feat_val_t>& feat_vals) {
    double sigma = 5;
    int w[] = {1,2,3,4,5,6, 12, 31, 8, 0};
    // int w[] = {2,3,7,9,-1,6, 12, 31, 8, 0};

    feat_idxs.resize(feature_cnt);
    feat_vals.resize(feature_cnt);

    // bias
    // feat_idxs[0] = 0;
    // feat_vals[0] = 1;
    for (int i = 0; i < feature_cnt; i++) {
        double v = rand_uniform();
        sigma += v * w[i];

        feat_idxs[i] = i;
        feat_vals[i] = v;
    }

    // feat_idxs[feature_cnt] = feature_cnt;
    // feat_vals[feature_cnt] = 1;
    return sigma;
}

void test(int num_iter, int freq) {
    int feature_cnt = 5;
    FTRLProximal ftrl(feature_cnt);

    srand(time(NULL));
    for (int i = 1; i <= num_iter; ++i) {
        // feature_vec_t feature_vec;
        // double y = mock_func(feature_cnt, feature_vec);
        // ftrl.fit(feature_vec, y);
        std::vector<feat_idx_t> feat_idxs;
        std::vector<feat_val_t> feat_vals;
        double y = mock_func(feature_cnt, feat_idxs, feat_vals);
        ftrl.fit(feat_idxs, feat_vals, y);

        if (i % freq == 0) {
            printf("iter=[%d] ", i);
            ftrl.dump_model();
        }
    }
    ftrl.dump_model();
}

void train() {
}

void predict() {
}

enum TaskType {
    kTrain, kPredict, kDumpModel
};

int main(int argc, char* argv[]) {
    // test(atoi(argv[1]), atoi(argv[2]));
    // return 0;
    // int feature_cnt = 152099;
    int feature_cnt = 153171;
    char* file_name = NULL;
    char* in_model_file = NULL;
    char* output_file = NULL;

    TaskType task_type = kTrain;
    int freq = 100;

    int c;
    const char* param = "t:r:f:i:o:h";
    while ((c = getopt(argc, argv, param)) != -1) {
        switch(c) {
            case 't':
                switch (atoi(optarg)) {
                    case kPredict:
                        task_type = kPredict;
                        fprintf(stderr, "Begin Predict...\n");
                        break;
                    case kDumpModel:
                        task_type = kDumpModel;
                        break;
                    default:
                        break;
                }
                break;
            case 'r':
                freq = atoi(optarg);
                break;
            case 'f':
                file_name = optarg;
                break;
            case 'i':
                in_model_file = optarg;
                break;
            case 'o':
                output_file = optarg;
                break;
            case 'h':
            case '?':
                fprintf(stderr, "Usage: %s -[%s]\n", argv[0], param);
                return 0;
        }
    }

    FTRLProximal ftrl(feature_cnt);
    if (in_model_file != NULL) {
        ftrl.load_model(in_model_file);
    }

    FILE* output_fp = NULL;
    if (task_type == kPredict) {
        output_fp = fopen(output_file, "w");
        if (output_fp == NULL) {
            fprintf(stderr, "open output_file [%s] error\n", output_file);
            return 1;
        }
    }

    FILE* fp = fopen(file_name, "rb");
    if (fp == NULL) {
        fprintf(stderr, "open data_file [%s] error\n", file_name);
        return 1;
    }

    int model_size;
    if (1 != fread(&model_size, sizeof(int), 1, fp)) {
        fprintf(stderr, "read model_size error\n");
    }

    double sum_error = 0;
    for (size_t i = 1; ; i++) {
        std::vector<feat_idx_t> feat_idxs(model_size);
        std::vector<feat_val_t> feat_vals(model_size);

        double y;
        if (1 != fread(&y, sizeof(double), 1, fp)) {
            if (feof(fp)) {
                fprintf(stderr, "[%lu] read eof\n", i);
            } else {
                fprintf(stderr, "read [%lu] label error\n", i);
            }
            break;
        }
        int read_bytes = static_cast<int>(fread(feat_idxs.data(), sizeof(int), model_size, fp));
        if (model_size != read_bytes) {
            fprintf(stderr, "read [%lu] feature_idxs error, need=[%d], read=[%d]\n", i, model_size, read_bytes);
            break;
        }
        read_bytes = static_cast<int>(fread(feat_vals.data(), sizeof(double), model_size, fp));
        if (model_size != read_bytes) {
            fprintf(stderr, "read [%lu] feature_vals error, need=[%d], read=[%d]\n", i, model_size, read_bytes);
            break;
        }

        if (task_type == kTrain) {
            double pred = ftrl.fit(feat_idxs, feat_vals, y);
            sum_error += (pred - y) * (pred - y);
            if (i % freq == 0) {
                printf("iter %lu: sum_error=[%f] mse=[%f]\n", i, sum_error, sum_error / freq);
                sum_error = 0;
            }
        }

        if (task_type == kPredict) {
            double pred = ftrl.predict(feat_idxs, feat_vals);
            double old_pred = 0;
            for (size_t i = 0; i < feat_vals.size(); i++) {
                old_pred += feat_vals[i];
            }

            fprintf(output_fp, "%f %f %f\n", y, old_pred, pred);
        }
    }
    fclose(fp);
    
    if (task_type == kPredict) {
        fclose(output_fp);
    }

    if (task_type == kTrain) {
        ftrl.dump_model(output_file, true);
    }
    return 0;
}
