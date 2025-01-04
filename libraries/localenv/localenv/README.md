## env-api

normalmap,light_point,view_pointを入力として，回転されたview_point(wo)とwiを返す．

（wiは無限遠を仮定するので実際は回転されていないことに注意）

normalmapの各法線ベクトルが[0,0,1]になるように回転し，それに合わせてwoも回転する

戻り値はwiとwo

