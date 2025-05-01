# Processo Context

Confronto tra le diverse fasi del processo di segmentazione.

| Passaggio            | Immagine           |
|----------------------|--------------------|
| Original Image       | ![][orig_img]      |
| Florence Bounding Boxes | ![][florence_bboxes] |
| Context Tiles        | ![][context_tiles] |
| Colored Masks        | ![][colored_masks] |
| Cleaned Mask         | ![][cleaned_mask]  |
| Final Mask           | ![][final_mask]    |

<!-- Alias immagini -->

[orig_img]: test_images/context_process/1_original%20image.png
[florence_bboxes]: test_images/context_process/1_florence%20bounding%20boxes.png
[context_tiles]: test_images/context_process/1_context%20tiles.png
[colored_masks]: test_images/context_process/1_colored%20masks.png
[cleaned_mask]: test_images/context_process/1_cleaned%20mask.png
[final_mask]: test_images/context_process/1_final%20mask.png

# Risultati dei test

Confronto visivo tra diversi metodi di segmentazione. Ogni riga mostra un soggetto, l'immagine originale, e i risultati ottenuti con sei varianti.

| Prompt        | Originale        | Florence         | Context          | Tiled            | Large            | Small            |
|---------------|------------------|------------------|------------------|------------------|------------------|------------------|
| golf ball     | ![][gb_orig]     | ![][gb_flor]     | ![][gb_ctx]      | ![][gb_tiled]    | ![][gb_large]    | ![][gb_small]    |
| golf club     | ![][gc_orig]     | ![][gc_flor]     | ![][gc_ctx]      | ![][gc_tiled]    | ![][gc_large]    | ![][gc_small]    |
| human face    | ![][hf_orig]     | ![][hf_flor]     | ![][hf_ctx]      | ![][hf_tiled]    | ![][hf_large]    | ![][hf_small]    |
| shoes, hands, cap, mouth, pants, ball, Nike logo, watch  | ![][mix_orig]    | ![][mix_flor]    | ![][mix_ctx]     | ![][mix_tiled]   | ![][mix_large]   | ![][mix_small]   |

<!-- Alias immagini -->

[gb_orig]: test_images/comparisons/1_original%20image.jpg
[gb_flor]: test_images/comparisons/1_golf%20ball_florence.jpg
[gb_ctx]: test_images/comparisons/1_golf%20ball_context.jpg
[gb_tiled]: test_images/comparisons/1_golf%20ball_tiled.jpg
[gb_large]: test_images/comparisons/1_golf%20ball_large.jpg
[gb_small]: test_images/comparisons/1_golf%20ball_small.jpg

[gc_orig]: test_images/comparisons/1_original%20image.jpg
[gc_flor]: test_images/comparisons/1_golf%20club_florence.jpg
[gc_ctx]: test_images/comparisons/1_golf%20club_context.jpg
[gc_tiled]: test_images/comparisons/1_golf%20club_tiled.jpg
[gc_large]: test_images/comparisons/1_golf%20club_large.jpg
[gc_small]: test_images/comparisons/1_golf%20club_small.jpg

[hf_orig]: test_images/comparisons/1_original%20image.jpg
[hf_flor]: test_images/comparisons/1_human%20face_florence.jpg
[hf_ctx]: test_images/comparisons/1_human%20face_context.jpg
[hf_tiled]: test_images/comparisons/1_human%20face_tiled.jpg
[hf_large]: test_images/comparisons/1_human%20face_large.jpg
[hf_small]: test_images/comparisons/1_human%20face_small.jpg

[mix_orig]: test_images/comparisons/2_original%20image.jpg
[mix_flor]: test_images/comparisons/2_shoes,%20hands,%20cap,%20mouth,%20pants,%20ball,%20Nike%20logo,%20watch_florence.jpg
[mix_ctx]: test_images/comparisons/2_shoes,%20hands,%20cap,%20mouth,%20pants,%20ball,%20Nike%20logo,%20watch_context.jpg
[mix_tiled]: test_images/comparisons/2_shoes,%20hands,%20cap,%20mouth,%20pants,%20ball,%20Nike%20logo,%20watch_tiled.jpg
[mix_large]: test_images/comparisons/2_shoes,%20hands,%20cap,%20mouth,%20pants,%20ball,%20Nike%20logo,%20watch_large.jpg
[mix_small]: test_images/comparisons/2_shoes,%20hands,%20cap,%20mouth,%20pants,%20ball,%20Nike%20logo,%20watch_small.jpg
