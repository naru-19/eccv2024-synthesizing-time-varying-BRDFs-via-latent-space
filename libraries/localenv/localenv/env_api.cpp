#include <bits/stdc++.h>
using namespace std;
vector<vector<float> > _dot(const vector<vector<float> > &A,
                            const vector<vector<float> > &B) {
  int a = A.size();
  int b = B.size();
  int c = B[0].size();
  // a x c 行列を生成
  vector<vector<float> > C(a, vector<float>(c));
  for (int i = 0; i < a; i++) {
    for (int j = 0; j < c; j++) {
      for (int k = 0; k < b; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return C;
}
// for debug
void print(float **norm, int i) {
  cout << norm[i][0] << ", " << norm[i][1] << ", " << norm[i][2] << endl;
}

extern "C" {
void env_from_normalmap(float **norm, float **local_wo, float **view,
                        int length) {
  vector<vector<float> > V = {{view[0][0]}, {view[1][0]}, {view[2][0]}};
  for (int i = 0; i < length; i++) {
    if (norm[i][2] <= 0) { /*見えない部分は0*/
      local_wo[i][0] = 0, local_wo[i][1] = 0, local_wo[i][2] = 0;
    } else { /*normを[0,0,1]に回転する行列を求め，その回転行列(rx,ry)を用いてviewを回転させる．*/
      // 回転角
      float x = atan2(norm[i][1], norm[i][2]);
      float y = atan2(-norm[i][0], hypot(norm[i][1], norm[i][2]));
      // 回転行列
      vector<vector<float> > rx = {
          {1, 0, 0}, {0, cos(x), -sin(x)}, {0, sin(x), cos(x)}};
      vector<vector<float> > ry = {
          {cos(y), 0, sin(y)}, {0, 1, 0}, {-sin(y), 0, cos(y)}};
      // viewの回転
      vector<vector<float> > wo = _dot(_dot(ry, rx), V);
      // 書き込み
      if (wo[2][0] <= 0) {
        wo[0][0] = 0, wo[1][0] = 0, wo[2][0] = 0;
      }
      local_wo[i][0] = wo[0][0], local_wo[i][1] = wo[1][0],
      local_wo[i][2] = wo[2][0];
    }
  }
}
}
