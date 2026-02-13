
from src.utils.logging import setup_logging
setup_logging()
import pyro
import torch  # noqa: E402
import numpy as np  # noqa: E402
from torch.utils.data import DataLoader
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from src.models.ours.ATCSkillModel import ATCSkillModel

import optuna  # noqa: E402
from src.adapters.ours.GamenetFastAdapter import GamenetFastAdapter, GamenetFastDataset # noqa: E402
from src.models.ours.GamenetFast import GamenetFast # noqa: E402
from src.adapters.original.GamenetAdapter import GamenetAdapter  # noqa: E402
from src.adapters.ours.KNNAdapter import KNNAdapter  # noqa: E402
from src.load.MIMIC3Loader import MIMIC3Loader, MIMIC3LoaderConfig  # noqa: E402
from src.models.original.GamenetOG import GameNetTrainConfig, GamenetOriginal  # noqa: E402
from src.models.ours.KNN import KNN  # noqa: E402
from src.models.ours.LinearModel import LinearModel  # noqa: E402
from src.preprocess.BasePreprocessor import ProcessedTables  # noqa: E402
from src.preprocess.MIMIC3Preprocessor import MIMIC3Preprocessor  # noqa: E402
from src.utils.metrics import evaluate_multilabel_sets  # noqa: E402
from src.utils.one_hot_encode import multihot_to_id_lists  # noqa: E402
from src.utils.stats import mimic3_dataset_stats  # noqa: E402
from src.adapters.original.MICRONAdapter import MICRONAdapter # noqa: E402
from src.models.original.Micron import MICRONOriginal # noqa: E402
from src.adapters.ours.LastVisistAdapter import LastVisitAdapter
from src.models.ours.LastVisit import LastVisit
from src.models.ours.MostPopular import MostPopular
from src.adapters.ours.MLPAdapter import MLPAdapter
from src.models.ours.MLP import MLPModel
from src.adapters.original.MoleRecAdapter import MoleRecAdapter
from src.models.original.MoleRec import MoleRecConfig, OriginalMoleRec
from src.adapters.ours.MultiHotAdapter import MultiHotAdapter, MultiHotDataset
from src.adapters.original.SafeDrugAdapter import SafeDrugAdapter
from src.models.original.SafeDrug import SafeDrugOriginal
from src.models.original.FastRx import FastRxOriginal
from src.models.original.LamRec import LamRecOriginal
from src.models.original.RareMed import RareMedOriginal

from src.models.original.SubRec import SubRecOriginal


def test_mimic3_loader_initialization():
    loader = MIMIC3Loader(cfg=MIMIC3LoaderConfig())
    response = loader.load()
    return response

def test_preprocessor(loaded_data):
    preprocessor = MIMIC3Preprocessor()
    data = preprocessor.process(loaded_data)
    return data

def jaccard(y_true, y_pred):
    return [
        len(set(t) & set(p)) / len(set(t) | set(p)) if (t or p) else 1.0
        for t, p in zip(y_true, y_pred)
    ]

def test_knn_training(data: ProcessedTables):
    adapter = KNNAdapter()
    X_train, y_train, X_val, y_val, X_test, y_test = adapter.adapt(data)
    
    knn = KNN(k=50, similarity="cosine", agg="softmax", threshold=0.3)
    knn.fit(X_train, y_train)
    pred, _ = knn.predict(X_test, return_scores=True)
    y_pred = multihot_to_id_lists(pred)
    metrics = evaluate_multilabel_sets(y_test, y_pred, data.artefacts["ddi_adj"], ignore_ids=[0, 1])
    print(metrics)

def test_linear_training(data: ProcessedTables):
    adapter = KNNAdapter()
    X_train, y_train, X_val, y_val, X_test, y_test = adapter.adapt(data)
    
    model = LinearModel(in_dim=X_train.size(1), out_dim=y_train.size(1), device=torch.device("cpu:0"))
    history = model.fit(X_train, y_train)
    pred, prob = model.predict(X_test)
    y_pred = multihot_to_id_lists(pred)
    metrics = evaluate_multilabel_sets(y_test, y_pred, data.artefacts["ddi_adj"], ignore_ids=[0, 1])
    print(metrics)
    
def test_gamenet_training(data: ProcessedTables):
    adapter = GamenetAdapter()
    train_samples, val_samples, test_samples, vocab_size = adapter.adapt(data)
    model = GamenetOriginal(cfg = GameNetTrainConfig(epochs=40), vocab_size=vocab_size, ehr_adj=data.artefacts["ehr_adj"], ddi_adj=data.artefacts["ddi_adj"])
    history = model.fit(train_samples=train_samples, val_samples=val_samples)

def test_micron_training(data: ProcessedTables):
    adapter = MICRONAdapter()
    train_samples, val_samples, test_samples, vocab_size = adapter.adapt(data)
    model = MICRONOriginal(vocab_size=vocab_size, ddi_adj=data.artefacts['ddi_adj'])
    history = model.fit(train_samples=train_samples, val_samples=val_samples)
    
def test_gamenet_test(data: ProcessedTables):
    adapter = GamenetAdapter()
    train_samples, val_samples, test_samples, vocab_size = adapter.adapt(data)
    model = GamenetOriginal(cfg = GameNetTrainConfig(epochs=40), vocab_size=vocab_size, ehr_adj=data.artefacts["ehr_adj"], ddi_adj=data.artefacts["ddi_adj"])
    model.load_model("saved/GAMENETOG/best_model.pt", weights_only=False)
    y_test, y_pred = [], []
    threshold = getattr(model.cfg, "threshold", 0.5)

    for prefix, target_meds in test_samples:
        y_test.append([m for m in target_meds if m != 0])

        pred_ids = model.predict(prefix, threshold=threshold, topk=None)
        pred_ids = [m for m in pred_ids if m != 0]
        y_pred.append(pred_ids)

    metrics = evaluate_multilabel_sets(
        y_test,
        y_pred,
        data.artefacts["ddi_adj"],
        ignore_ids=[0, 1],
    )
    print(metrics)
    
    
def test_micron_test(data: ProcessedTables):
    adapter = MICRONAdapter()
    train_samples, val_samples, test_samples, vocab_size = adapter.adapt(data)
    model = MICRONOriginal(vocab_size=vocab_size, ddi_adj=data.artefacts['ddi_adj'])
    model.load_model("saved/MICRONOG/best_model.pt", weights_only=False)
    y_test, y_pred = [], []
    threshold = getattr(model.cfg, "threshold", 0.5)

    for prefix, target_meds in test_samples:
        y_test.append([m for m in target_meds if m != 0])

        pred_ids = model.predict(prefix, threshold=threshold, topk=None)
        pred_ids = [m for m in pred_ids if m != 0]
        y_pred.append(pred_ids)

    metrics = evaluate_multilabel_sets(
        y_test,
        y_pred,
        data.artefacts["ddi_adj"],
        ignore_ids=[0, 1],
    )
    print(metrics)

def test_gamenet_fast_train(data: ProcessedTables):
    batch_size = 64
    adapter = GamenetFastAdapter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test, vocab_size, ehr_adj, ddi_adj = adapter.adapt(data)
    train_dataset, val_dataset, test_dataset = GamenetFastDataset(train, device), GamenetFastDataset(val, device), GamenetFastDataset(test, device),
    train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=True), 
    model = GamenetFast(GameNetTrainConfig(weight_decay=1e-4, batch_size=batch_size), vocab_size, ehr_adj, ddi_adj, device=device)
    model.fit(train_loader, val_loader)

def test_last_visit(data: ProcessedTables):
    adapter = LastVisitAdapter()
    train, val, test, vocab_size = adapter.adapt(data)
    model = LastVisit(vocab_size, data.artefacts["ddi_adj"])
    val_history = model.fit(train, val)
    test_history = model.predict(test)
    
def test_most_pop(data: ProcessedTables):
    adapter = LastVisitAdapter()
    train, val, test, vocab_size = adapter.adapt(data)
    model = MostPopular(vocab_size, data.artefacts["ddi_adj"])
    val_history = model.fit(train, val)
    test_history = model.predict(test)
    
def test_mlp(data: ProcessedTables):
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = MLPAdapter()
    train_loader, val_loader, test_loader, vocab_size = adapter.adapt(data, device, batch_size)
    model = MLPModel(vocab_size, device, num_layers=3)
    model.fit(train_loader, val_loader)
    
def test_molerec(data: ProcessedTables):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = MoleRecAdapter()
    train_samples, val_samples, test_samples, vocab_size, average_projection, molecule_graph, substruct_graphs, ddi_mask_H = adapter.adapt(data)
    model = OriginalMoleRec(
        cfg=MoleRecConfig(), 
        ddi_adj=data.artefacts["ddi_adj"], 
        voc_size=vocab_size, 
        molecule_graph=molecule_graph, 
        average_projection=average_projection,
        substruct_data=substruct_graphs,
        ddi_mask_H=ddi_mask_H,
        device=device
    )
    model.fit(train_samples, val_samples)
    
def test_atc_skill(data: ProcessedTables):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = MultiHotAdapter()
    batch_size=256
    train, val, test, vocab_size = adapter.adapt(data)
    train_dataset, val_dataset, test_dataset = MultiHotDataset(train, device), MultiHotDataset(val, device), MultiHotDataset(test, device)
    train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    model_inst = ATCSkillModel(n_features=vocab_size[0], n_drugs=vocab_size[2]).to(device)
    # 2. Setup SVI with Adam
    # We use a slightly higher learning rate for the means than the sigmas usually
    optimizer = Adam({"lr": 0.005})
    svi = SVI(model_inst.model, model_inst.guide, optimizer, loss=Trace_ELBO())

    # 3. Training Loop
    num_epochs = 100
    batch_size = 256 # GPU loves powers of
    for epoch in range(num_epochs):
        epoch_loss = 0
        # Use a standard PyTorch DataLoader for your 5k-dim inputs
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # This one line handles the forward pass, KL penalty, 
            # gradient calculation, and parameter update
            loss = svi.step(x_batch, y_batch)
            epoch_loss += loss

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | ELBO Loss: {epoch_loss / len(train_loader.dataset)}")
    
    learned_rules = pyro.param("mi").detach().cpu().numpy()
    vocab = data.artefacts["vocab"]
    # Find the most 'mutually exclusive' drugs
    min_val = np.min(learned_rules)
    idx = np.unravel_index(np.argmin(learned_rules), learned_rules.shape)
    print(f"The model learned that {vocab['med'][1][int(idx[0])]} and {vocab['med'][1][idx[1]]} almost never go together.")


def test_safedrug(data: ProcessedTables):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = SafeDrugAdapter()
    train_samples, val_samples, test_samples, vocab_size, ddi_mask_H, MPNNSet, n_fingerprint, average_projection = adapter.adapt(data)
    model = SafeDrugOriginal(vocab_size, data.artefacts["ddi_adj"], device, ddi_mask_H, MPNNSet, n_fingerprint, average_projection)
    val_history = model.fit(train_samples, val_samples)
    
def test_fastrx_training(data: ProcessedTables):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = GamenetAdapter()
    train_samples, val_samples, test_samples, vocab_size = adapter.adapt(data)
    model = FastRxOriginal(
        vocab_size=vocab_size, 
        device=device, 
        ehr_adj=data.artefacts["ehr_adj"], 
        ddi_adj=data.artefacts["ddi_adj"],
    )
    history = model.fit(train_samples, val_samples)
    
def test_LamRec_training(data: ProcessedTables):
    batch_size = 64
    adapter = GamenetFastAdapter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test, vocab_size, ehr_adj, ddi_adj = adapter.adapt(data)
    train_dataset, val_dataset, test_dataset = GamenetFastDataset(train, device), GamenetFastDataset(val, device), GamenetFastDataset(test, device),
    train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(val_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=True), 
    model = LamRecOriginal(vocab_size=vocab_size, ddi_adj=ddi_adj, device=device)
    model.fit(train_loader, val_loader)
    
def test_RAREMed_training(data: ProcessedTables):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = GamenetAdapter()
    train_samples, val_samples, test_samples, vocab_size = adapter.adapt(data)
    model = RareMedOriginal(
        vocab_size=vocab_size,
        ddi_adj=data.artefacts["ddi_adj"], 
        device=device,
    )
    history = model.fit(train_samples, val_samples)
    
def test_subrec_training(data: ProcessedTables):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adapter = MoleRecAdapter()
    train_samples, val_samples, test_samples, vocab_size, average_projection, molecule_graph, substruct_graphs, ddi_mask_H = adapter.adapt(data)
    model = SubRecOriginal(vocab_size=vocab_size, device=device, ddi_adj=data.artefacts["ddi_adj"])
    history = model.fit(train_samples, molecule_graph, average_projection, val_samples)
    
response = test_mimic3_loader_initialization()
data = test_preprocessor(response)
# out = mimic3_dataset_stats(data)
#test_knn_training(data)
#test_linear_training(data)
#test_gamenet_training(data)
#test_micron_training(data)
#test_gamenet_test(data)
#test_micron_test(data)
#test_gamenet_fast_train(data)
#test_last_visit(data)
# test_most_pop(data)
# test_mlp(data)
#test_molerec(data)
#test_atc_skill(data)
# test_safedrug(data)
# test_fastrx_training(data)
# test_LamRec_training(data)
# test_RAREMed_training(data)
test_subrec_training(data)