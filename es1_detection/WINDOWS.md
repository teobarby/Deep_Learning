# Addestramento su Windows con GPU NVIDIA

Il setup completo è nel [README](README.md); qui restano solo le note specifiche di Windows.

## PyTorch con CUDA

```powershell
cd es1_detection
py -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Verifica (deve stampare `True` e il nome della GPU):
```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Se stampa `False` è stata installata la versione solo CPU: disinstallare torch e
reinstallarlo con `--index-url` come sopra.

## Durante il training

- Il training completo (30 epoche) dura circa **un'ora** su una RTX 4070 Ti e usa
  ~3,5 GB di memoria video. In caso di *out of memory*, ridurre il batch:
  `python src\train.py --batch_size 8`.
- Impostare la sospensione del PC su "Mai": un run interrotto si riprende con
  `python src\train.py --resume`, ma riparte dall'ultima epoca salvata.
- **Smart App Control** può bloccare il caricamento delle DLL di PyTorch
  (`An Application Control policy has blocked this file`). In tal caso va disattivato
  dalle impostazioni di sicurezza di Windows, oppure si usa WSL2.
- Se il DataLoader dà errori di multiprocessing, mettere `num_workers=0` nei
  `DataLoader` di `src/train.py`.
- Non modificare i file in `src/` mentre un training è in corso: i worker del
  DataLoader li rileggono a ogni epoca. Per lanciare più esperimenti in sequenza,
  `tools\run_queue.ps1` ne congela una copia per ciascuno.

## Portare il modello su un'altra macchina

Basta copiare `outputs\<esperimento>\best.pt`: il checkpoint contiene i pesi **e** la
configurazione con cui sono stati addestrati, che viene verificata al caricamento
(`config.check_checkpoint_config`). Se `config.py` non corrisponde, il caricamento si
interrompe con un errore invece di produrre risultati sbagliati in silenzio.
