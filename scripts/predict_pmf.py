"""PMF inference on one flat sample or a prepared test2 batch."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import torch
from easydict import EasyDict
from MMSA.models import AMIO


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--weights",type=Path,required=True)
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--output",type=Path)
    opt=parser.parse_args()
    args=EasyDict(json.loads(opt.config.read_text(encoding="utf-8")))
    if args.model_name != "pmf":
        parser.error("Expected a PMF configuration")
    args.device=torch.device(opt.device)
    with opt.features.open("rb") as stream:
        data=pickle.load(stream)
    data=data.get("test2",data)
    missing={"text","text_bert","audio","vision"}-data.keys()
    if missing:
        parser.error(f"Missing fields {sorted(missing)}; build test2 first for attachment 3")
    arrays={key:np.asarray(data[key],dtype=np.float32) for key in ("text","text_bert","audio","vision")}
    if arrays["text"].ndim==2:
        arrays={key:value[None] for key,value in arrays.items()}
    count=len(arrays["text"])
    ids=np.asarray(data.get("id",[str(i) for i in range(count)])).reshape(-1)
    if len(ids)!=count:
        parser.error("ID count differs from sample count")
    model=AMIO(args).to(args.device)
    model.load_state_dict(torch.load(opt.weights,map_location=args.device,weights_only=True))
    model.eval()
    rows=[]
    with torch.no_grad():
        for start in range(0,count,args.batch_size):
            b={key:torch.as_tensor(value[start:start+args.batch_size],device=args.device) for key,value in arrays.items()}
            out=model(b["text"],b["audio"],b["vision"],padding_mask=b["text_bert"][:,1,:])
            classes=out["polarity"].argmax(-1).cpu().tolist()
            magnitudes=out["magnitude"].cpu().tolist()
            values=out["M"].flatten().cpu().tolist()
            for offset,(cls,mag,value) in enumerate(zip(classes,magnitudes,values)):
                rows.append({"id":str(ids[start+offset]),"polarity":cls,"polarity_name":["negative","neutral","positive"][cls],"magnitude":mag,"intensity":value})
    text=json.dumps(rows,ensure_ascii=False,indent=2)+"\n"
    if opt.output:
        opt.output.parent.mkdir(parents=True,exist_ok=True)
        opt.output.write_text(text,encoding="utf-8")
    else:
        print(text,end="")


if __name__=="__main__":
    main()
