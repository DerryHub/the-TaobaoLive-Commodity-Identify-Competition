import argparse

def get_args_efficientdet():
    parser = argparse.ArgumentParser("EfficientDet")
    parser.add_argument("--image_size", type=int, default=512, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=128, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--data_path", type=str, default="tcdata/test_dataset_fs", help="the root folder of dataset")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument('--network', default='efficientdet-d4', type=str,
                        help='efficientdet-[d0, d1, ..]')
    parser.add_argument("--is_training", type=bool, default=True)
    parser.add_argument('--resume', type=bool, default=True)

    parser.add_argument('--nms_threshold', type=float, default=0.3)
    parser.add_argument("--cls_threshold", type=float, default=0.3)
    parser.add_argument('--cls_2_threshold', type=float, default=0.5)
    parser.add_argument('--iou_threshold', type=float, default=0.4)
    parser.add_argument('--instance_threshold_image', type=float, default=0.35)
    parser.add_argument('--instance_threshold_video', type=float, default=0.35)
    parser.add_argument('--prediction_dir', type=str, default="predictions/")
    parser.add_argument("--workers", type=int, default=48)

    parser.add_argument("--imgORvdo", type=str, default=None, help='[image, video]')

    # BERT config
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_hidden_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=512)
    parser.add_argument("--hidden_act", type=str, default='gelu')
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--max_position_embeddings", type=int, default=64)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--layer_norm_eps", type=int, default=1e-12)
    parser.add_argument("--output_size", type=int, default=None)
    parser.add_argument("--PAD", type=int, default=0)
    args = parser.parse_args()
    return args

def get_args_arcface():
    parser = argparse.ArgumentParser("ArcFace")
    parser.add_argument("--size", type=int, default=256, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=256, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--data_path", type=str, default="tcdata/test_dataset_fs", help="the root folder of dataset")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--num_classes", type=int, default=29522)
    parser.add_argument("--drop_ratio", type=float, default=0.1)
    parser.add_argument("--embedding_size", type=int, default=2048)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument("--s", type=float, default=64.0)
    parser.add_argument("--m", type=float, default=0.5)
    
    parser.add_argument("--network", type=str, default='resnet', 
                        help="[resnest, resnet, googlenet, inceptionv4, inceptionresnetv2, resnet_cbam]")

    parser.add_argument("--validation_config", type=list,
                        default=[('resnest', 50, 1, ''), ('resnest', 101, 1.3, ''), ('resnet_cbam', 101, 1, '')])

    # resnest config
    parser.add_argument("--num_layers_s", type=int, default=50, help="[50, 101]")

    # resnet config
    parser.add_argument("--num_layers_r", type=int, default=50, help="[50, 100, 152]")
    parser.add_argument("--mode", type=str, default='ir_se', help="[ir, ir_se]")

    # resnet cbam config
    parser.add_argument("--num_layers_c", type=int, default=101, help="[50, 101, 152]")

    # googlenet config

    # inceptionv4 config

    # inceptionresnetv2 config

    # densenet config
    parser.add_argument("--num_layers_d", type=int, default=121, help="[121, 161, 169, 201]")

    args = parser.parse_args()
    return args

def get_args_classifier():
    parser = argparse.ArgumentParser("Classifier")
    parser.add_argument("--image_size", type=int, default=112, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=256, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--data_path", type=str, default="data", help="the root folder of dataset")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument('--network', default='efficientnet-b0', type=str,
                        help='efficientnet-[b0, b1, ..]')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    return args
