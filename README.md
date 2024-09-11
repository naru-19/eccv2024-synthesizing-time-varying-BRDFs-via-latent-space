# eccv2024-synthesizing-time-varying-BRDFs-via-latent-space

## setup
```
docker build --build-arg USERNAME=${username} \
       --build-arg UID=${uid} \
       --build-arg GROUPNAME=${groupname} \
       --build-arg GID=${gid} \
       -t repo-luna.ist.osaka-u.ac.jp:5000/${username}/${imagename}:${tag} .

# docker push repo-luna.ist.osaka-u.ac.jp:5000/narumoto/latent_cuda102:latent_cuda102
```