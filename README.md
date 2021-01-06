# video_captioning
普通参数：
1. epoch_num: number of epochs for training
2. save_per_epoch: 每个epoch做多少次evaluation
3. train_batch_size
4. test_batch_size
5. use_glove: 使用glove pre-train 的 word embedding

GNN 参数：
1. num_obj：每帧使用多少个 region features (1-36)
2. num_proposals：number of proposals for both object and motion

GAN 参数：
1. use_visual_gan：使用conditional discriminator
2. use_lang_gan： 使用 text discriminator
3. num_D_lang: train 一次 G， train num_D_lang 次 text discriminator
4. num_D_visual: train 一次 G， train num_D_visual 次 conditional discriminator
