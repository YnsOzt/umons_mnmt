from .sat import ShowAttendAndTell
from .nmt import NMT
from .amnmt import AttentiveMNMT
from .amnmtfeats import AttentiveMNMTFeatures
from .mnmtdecinit import MNMTDecinit
from .amnmt_trgmul import AttentiveMNMTFeaturesTRGMUL
from .amnmt_ctxmul import AttentiveMNMTFeaturesCTXMUL
from .amnmt_sa_sga import AttentiveMNMTFeaturesSASGA
from .amnmt_sa_sga_trgmul import AttentiveMNMTFeaturesSASGATRGMUL
from .amnmt_sa_trgmul import AttentiveMNMTFeaturesSATRGMUL
from .amnmt_txt_sa import AttentiveMNMTFeaturesTXTSA
from .VMT import VMT
from .IMT import IMT
from .VIC import VIC

# Speech models
from .asr import ASR

# Experimental: requires work/adaptation
from .multitask import Multitask
from .multitask_att import MultitaskAtt

##########################################
# Backward-compatibility with older models
##########################################
ASRv2 = ASR
