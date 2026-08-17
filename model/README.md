# Model artifact

Train the baseline and place the resulting artifact here as
`retrieval_model.npz` before building a submission image:

```bash
cbct-reasoner train --data /path/to/toothfairy4 --output model/retrieval_model.npz
```

The artifact contains derived global image features and selected report text. Treat it
as clinical data: verify the dataset terms permit redistribution before publishing it.
No weights or protected patient data are committed to this repository.
