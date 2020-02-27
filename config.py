import argparse

def get_args_efficientdet():
    parser = argparse.ArgumentParser("EfficientDet")
    parser.add_argument("--image_size", type=int, default=512, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=8, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument('--alpha', type=float, default=0.25)
    parser.add_argument('--gamma', type=float, default=1.5)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=0,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("--data_path", type=str, default="data", help="the root folder of dataset")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument('--network', default='efficientdet-d0', type=str,
                        help='efficientdet-[d0, d1, ..]')
    parser.add_argument("--is_training", type=bool, default=True)
    parser.add_argument('--resume', type=bool, default=False)

    parser.add_argument('--nms_threshold', type=float, default=0.3)
    parser.add_argument("--cls_threshold", type=float, default=0.3)
    parser.add_argument('--cls_2_threshold', type=float, default=0.5)
    parser.add_argument('--iou_threshold', type=float, default=0.4)
    parser.add_argument('--prediction_dir', type=str, default="predictions/")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    return args

def get_args_arcface():
    parser = argparse.ArgumentParser("ArcFace")
    parser.add_argument("--image_size", type=int, default=128, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=64, help="The number of images per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--data_path", type=str, default="data", help="the root folder of dataset")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=50, help="[50, 100, 152]")
    parser.add_argument("--drop_ratio", type=float, default=0.1)
    parser.add_argument("--mode", type=str, default='ir', help="[ir, ir_se]")
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument("--workers", type=int, default=2)
    
    args = parser.parse_args()
    return args
