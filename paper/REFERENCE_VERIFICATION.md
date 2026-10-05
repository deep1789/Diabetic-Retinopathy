# Reference verification report

Every entry of `paper/refs.bib` was checked against an external record. Method: (1) automatic Crossref matching on title, first author, year, volume and first page (script `code/verify_refs.py`, raw output `paper/results/refcheck.json`); (2) direct DOI lookup for unmatched or mismatched items; (3) publisher/proceedings pages (PMLR, NeurIPS, CVF, OpenReview) and arXiv records for conference papers; (4) web search for items without DOI.

Result: **7 page-number errors found and corrected**, 2 unverifiable page ranges removed, 3 unused entries deleted, 20 recent (2023-2026) references added and verified. Pages and article numbers are those of the cited record; author lists with "and others" are truncated by design.

| Key | Verification | Outcome |
|---|---|---|
| `li2019ddr` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `porwal2020idrid` | DOI lookup (Crossref 10.1016/j.media.2019.101561) | verified, no change |
| `zhou2021fgadr` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `gulshan2016` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `ting2017` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `abramoff2018` | DOI lookup (Crossref 10.1038/s41746-018-0040-6) | verified, no change |
| `teo2021` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `saeedi2019` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `wilkinson2003` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `etdrs1991` | DOI lookup (Crossref 10.1016/S0161-6420(13)38012-9) | verified: Ophthalmology 98(5):786-806; group author |
| `ronneberger2015` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `zhou2020unetpp` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `oktay2018` | OpenReview / web search (MIDL 2018) | verified title, authors, venue; no pages (MIDL) |
| `chen2018deeplab` | DOI lookup (Crossref 10.1007/978-3-030-01234-2_49) | CORRECTED pages 801-818 -> 833-851 |
| `he2016resnet` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `liu2022convnext` | DOI lookup (Crossref 10.1109/CVPR52688.2022.01167) | CORRECTED pages 11976-11986 -> 11966-11976 |
| `liu2021swin` | DOI lookup (Crossref 10.1109/ICCV48922.2021.00986) | CORRECTED pages 10012-10022 -> 9992-10002 |
| `dosovitskiy2021vit` | arXiv 2010.11929 record | verified title and authors; ICLR 2021 |
| `tan2019efficientnet` | PMLR page v97/tan19a | verified: ICML 2019, pp. 6105-6114 |
| `zhou2023retfound` | DOI lookup (Crossref 10.1038/s41586-023-06555-x) | verified: Nature 622(7981):156-163 |
| `ren2017fasterrcnn` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `tian2019fcos` | DOI lookup (Crossref 10.1109/ICCV.2019.00972) | CORRECTED pages 9627-9636 -> 9626-9635 |
| `lin2017focal` | DOI lookup (Crossref 10.1109/ICCV.2017.324) | CORRECTED pages 2980-2988 -> 2999-3007 |
| `salehi2017tversky` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `abraham2019ftl` | DOI lookup (Crossref 10.1109/ISBI.2019.8759329) | verified: ISBI 2019, pp. 683-687 |
| `graham2015kaggle` | web search (cited widely; Univ. of Warwick report, 2015) | exists; technical report without DOI |
| `aptos2019` | Kaggle competition page (HTTP 200) | verified existence |
| `rath2020gaussian` | Kaggle dataset API (title, creator, licence CC0) | verified |
| `decenciere2013teleophta` | Crossref match | verified (accent encoding differs only) |
| `he2021cabnet` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `li2020canet` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `wang2017zoomin` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `yang2017twostage` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `foo2020multitask` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `playout2019` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `huang2021lesioncl` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `sun2021lat` | DOI lookup (Crossref 10.1109/CVPR46437.2021.01079) | CORRECTED pages 10938-10947 -> 10933-10942 |
| `lin2018antinoise` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `quellec2017` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `zhou2019collab` | DOI lookup (Crossref 10.1109/CVPR.2019.00218) | CORRECTED pages 2079-2088 -> 2074-2083 |
| `niu2016ordinal` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `cao2020coral` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `shi2023corn` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `xu2018semanticloss` | PMLR page v80/xu18h | verified: ICML 2018, pp. 5502-5511 |
| `diligenti2017sbr` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `hu2016logic` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `vovk2005` | Crossref (Springer book) | verified: Springer 2005 |
| `angelopoulos2023` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `guo2017calibration` | PMLR page v70/guo17a | verified: ICML 2017, pp. 1321-1330 |
| `gal2016dropout` | PMLR page v48/gal16 | verified: ICML 2016, pp. 1050-1059 |
| `tarvainen2017meanteacher` | NeurIPS proceedings page | venue/year verified; unverifiable pages REMOVED |
| `ilse2018attentionmil` | PMLR page v80/ilse18a | verified: ICML 2018, pp. 2127-2136 |
| `loshchilov2019adamw` | OpenReview / web search | verified: ICLR 2019 |
| `paszke2019pytorch` | NeurIPS proceedings page | venue/year verified; unverifiable pages REMOVED |
| `wightman2019timm` | installed package metadata (author Ross Wightman) | repository page not reachable (403); package verified |
| `cohen1968kappa` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `efron1993bootstrap` | Crossref (CRC reprint 1994) | verified book; original Chapman & Hall 1993 |
| `padilla2021detmetrics` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `dice1945` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `saito2015pr` | Crossref match (title, first author, year, volume, first page) | verified, no change |
| `moor2023generalist` | Crossref | verified |
| `qiu2024visionfm` | Crossref + web search | verified: NEJM AI 1(12), AIoa2300221 |
| `silva2025flair` | Crossref | verified |
| `ma2024medsam` | Crossref DOI | verified: Nat Commun 15:654 |
| `kirillov2023sam` | Crossref | verified: ICCV 2023 pp. 3992-4003 |
| `woo2023convnextv2` | Crossref | verified |
| `oquab2024dinov2` | arXiv 2304.07193 + web search | verified: TMLR 2024 |
| `zhao2024rtdetr` | Crossref + CVF page | verified |
| `angelopoulos2024crc` | arXiv 2208.02814 + web search | verified: ICLR 2024 |
| `isensee2024nnunetrev` | Crossref | verified |
| `chen2024transunet` | Crossref | verified |
| `liu2023crosslesion` | Crossref | verified |
| `yue2023cascaded` | Crossref | verified |
| `lin2025cfhan` | Crossref | verified |
| `ikram2025resvit` | Crossref | verified |
| `li2026wfdenet` | Crossref | verified |
| `li2024wsrfnet` | Crossref | verified |
| `siebert2023dkl` | Crossref | verified |
| `hasan2026ordinalcp` | Crossref + publisher page via web search | verified |
| `vijayalakshmi2025multitask` | Crossref | verified |
