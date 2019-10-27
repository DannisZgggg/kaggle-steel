from data import *
import warnings
#from model.TGS_model import Res18Unetv5
from model.seresnet50 import Unet
from criterion import predict
import torch.nn.functional as F
from inference import post_process


def dice_multi_minsize(pred,mask,threshold,minsizes):
    #minsizes = np.arange(0, 4001, 200) #num_size
    pred = pred.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    assert pred.shape == mask.shape

    num_minsizes = minsizes.size
    num_classes = pred.shape[1]
    batch_size = pred.shape[0]

    dice = np.zeros((num_classes,num_minsizes))
    sigma = 1

    for b in range(batch_size):
        for c in range(num_classes):
            pred_layer = pred[b][c]#.astype(np.float32)
            mask_layer = mask[b][c].astype(np.float32)
            for min_idx,minsize in enumerate(minsizes):
                pred_layer_post, _ = post_process(pred_layer,threshold,minsize)
                pred_layer_post = pred_layer_post.astype(np.float32)
                dice[c][min_idx]+=\
                    (2*(pred_layer_post*mask_layer).sum()+sigma) / (pred_layer_post.sum() + mask_layer.sum()+sigma)
    return dice/batch_size



if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    seed = 69
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    best_threshold = 0.5
    minsizes = np.arange(0, 4001, 200) #num_size
    multi_dice = np.ones((1,4,minsizes.size))


    dataloader = provider(
        data_folder="input/severstal-steel-defect-detection/",
        df_path='input/severstal-steel-defect-detection/train.csv',
        phase='val',
        mean=(0.485, 0.456, 0.406),  # (0.39, 0.39, 0.39),
        std=(0.229, 0.224, 0.225),
        batch_size=1,
        num_workers=6,
    )

    ckpt_path = "weights/onlydefectRAdamsteppytorchBCE1018.pth"
    device = torch.device("cuda")
    model = Unet('se_resnet50',encoder_weights=None,classes=4,activation='sigmoid').to(device)
    model.eval()
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    state = state["state_dict"]
    for k in list(state.keys()):
        if 'module' == k.split('.')[0]:
            state[k[7:]] = state.pop(k)
    model.load_state_dict(state)
    print("Model Loaded!!!")


    # start prediction
    predictions = []
    for i, batch in enumerate(tqdm(dataloader)):
        img,mask = batch #[n,c,h,w]
        batch_preds = model(img.to(device))
        batch_preds = F.sigmoid(batch_preds)#[n,c,h,w]
        #batch_preds = predict(batch_preds,best_threshold)
        dice_group = dice_multi_minsize(batch_preds,mask,best_threshold,minsizes)
        dice_group = np.expand_dims(dice_group, axis=0)
        multi_dice = np.concatenate((multi_dice, dice_group), axis=0)

    dice_res = np.mean(multi_dice, axis=0) #[num_class,num_minsize]
    dice_res = dice_res.T #[num_minsize, num_class]
    df = pd.DataFrame(dice_res,index=np.arange(0, 4001, 200),columns=np.arange(1,5))
    df.to_csv("multi_dice.csv")
