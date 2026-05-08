TheThirty-EighthAAAIConferenceonArtificialIntelligence(AAAI-24)
|     |     |        | A        | Transfer |     | Approach | Using         | Graph |          |     |     |     |     |     |
| --- | --- | ------ | -------- | -------- | --- | -------- | ------------- | ----- | -------- | --- | --- | --- | --- | --- |
|     |     | Neural | Networks |          |     | in Deep  | Reinforcement |       | Learning |     |     |     |     |     |
TianpeiYang1,2*,HengYou1,JianyeHao1*,YanZheng1,MatthewE.Taylor2,3
1CollegeofIntelligenceandComputing,TianjinUniversity,
2UniveristyofAlberta,
3AlbertaMachineIntelligenceInstitute(Amii)
{tpyang,hengyou,yanzheng,jianye.hao}@tju.edu.cn
matthew.e.taylor@ualberta.ca
Abstract space as the target task, which severely limits generaliza-
tiontomorerealisticscenarioswherestate-actionspacemis-
Transferlearning(TL)hasshowngreatpotentialtoimprove
matchusuallyexists,whichwecallitcross-domainsetting.
ReinforcementLearning(RL)efficiencybyleveragingprior
Recently,severalapproacheshaveexploredcross-domain
knowledgeinnewtasks.However,muchoftheexistingTL
|          |            |              |           |     |         |       | TL from | the following |     | directions. | Some | works | align | states |
| -------- | ---------- | ------------ | --------- | --- | ------- | ----- | ------- | ------------- | --- | ----------- | ---- | ----- | ----- | ------ |
| research | focuses on | transferring | knowledge |     | between | tasks |         |               |     |             |      |       |       |        |
inacommonfeaturespacetocombatthemismatchusinga
thatsharethesamestate-actionspaces.Further,transferfrom
stateencoder.Totrainthestateencoder,Guptaetal.(2017)
| multiple | source tasks | that | have different | state-action |     | spaces |     |     |     |     |     |     |     |     |
| -------- | ------------ | ---- | -------------- | ------------ | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
is more challenging and needs to be solved urgently to im- requiretocollectpaireddataoftwotasksusingpre-trained
prove the generalization and practicality of the method in policies based on time alignment. However, it is expensive
real-worldscenarios.ThispaperproposesTURRET(Transfer and infeasible in real-world problems that the two agents
UsinggRaphneuRalnETworks),toutilizethegeneralization will perform at roughly the same rate. Later, Wan et al.
capabilities of Graph Neural Networks (GNNs) to facilitate (2020) train the state encoder using mutual information to
efficientandeffectivemulti-sourcepolicytransferlearningin
ensureahighcorrelationbetweenthestateembeddingsand
thestate-actionmismatchsetting.TURRETlearnsasemantic
|     |     |     |     |     |     |     | current | states. However, |     | they | ignore capturing |     | the | dynamic |
| --- | --- | --- | --- | --- | --- | --- | ------- | ---------------- | --- | ---- | ---------------- | --- | --- | ------- |
representationbyaccountingfortheintrinsicpropertyofthe
|     |     |     |     |     |     |     | information | of  | the environment |     | which | ultimately |     | leads to |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | --------------- | --- | ----- | ---------- | --- | -------- |
agentthroughGNNs,whichleadstoaunifiedstateembed-
|     |     |     |     |     |     |     | insufficient | transfer | performance. |     | Some | other | works | learn |
| --- | --- | --- | --- | --- | --- | --- | ------------ | -------- | ------------ | --- | ---- | ----- | ----- | ----- |
dingspaceforalltasks.Asaresult,TURRETachievesmore
|           |               |        |                |     |                 |     | both the | state and | action | mappings | for | transfer. | For | exam- |
| --------- | ------------- | ------ | -------------- | --- | --------------- | --- | -------- | --------- | ------ | -------- | --- | --------- | --- | ----- |
| efficient | transfer with | strong | generalization |     | ability between |     |          |           |        |          |     |           |     |       |
ple,Chenetal.(2019)capturethesemanticsofactionsus-
differenttasksandcanbeeasilycombinedwithexistingDeep
RLalgorithms.ExperimentalresultsshowthatTURRETsig- ing the effects on the environment, which is only applica-
nificantly outperforms other TL methods on multiple con- ble in discrete-action scenarios. Later, Zhang et al. (2021)
tinuousactioncontroltasks,successfullytransferringacross align the environment dynamics using a cycle consistency
robotswithdifferentstate-actionspaces. constraint. However, this method directly reuses the pre-
|     |     |     |     |     |     |     | trained | source policy | on  | the | target task | through | the | map- |
| --- | --- | --- | --- | --- | --- | --- | ------- | ------------- | --- | --- | ----------- | ------- | --- | ---- |
pinginazero-shorttransfermanner,whichmaynotachieve
Introduction
optimalperformance.Furthermore,alltheseabovemethods
|     |     |     |     |     |     |     | do not supporttransferring |     |     | from | multiple | source | tasks. | Re- |
| --- | --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | ---- | -------- | ------ | ------ | --- |
DeepReinforcementLearning(DRL)hasobtainedimpres-
|                |            |     |         |      |          |       | cently, CAT | (You | et al. | 2022) | has initially | done | this, | but it |
| -------------- | ---------- | --- | ------- | ---- | -------- | ----- | ----------- | ---- | ------ | ----- | ------------- | ---- | ----- | ------ |
| sive successes | in various |     | domains | such | as video | games |             |      |        |       |               |      |       |        |
failstohandlewhenthenumberofsourcepolicieschanges
| (Mnih et | al. 2015; | Silver | et al. 2016) | and | robotics | con- |     |     |     |     |     |     |     |     |
| -------- | --------- | ------ | ------------ | --- | -------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
trol (Lillicrap et al. 2016). However, DRL still faces the duetoitslimitationoflearningtheone-to-onemappingbe-
sample inefficiency problem, requiring considerable envi- tween each source task and the target task. Since previous
ronmentalinteractions.TransferLearning(TL)hasemerged works didn’t solve this problem properly, we argue a suit-
ablemethodthatcanhandlethemulti-sourcecross-domain
asapromisingtechniquetosignificantlyreduceDRLsam-
|                |               |     |                 |     |         |     | TL problem | and | is suitable | for | varying | numbers | of  | source |
| -------------- | ------------- | --- | --------------- | --- | ------- | --- | ---------- | --- | ----------- | --- | ------- | ------- | --- | ------ |
| ple complexity | by leveraging |     | prior knowledge |     | (Taylor | and |            |     |             |     |         |         |     |        |
Stone 2009; Zhu, Lin, and Zhou 2020; Yang et al. 2020a, tasksisurgentlyneededtoimprovethegeneralizabilityand
2021). Policy transfer is one major class of RL transfer practicality. Moreover, all the above methods use a multi-
methods, that focuses on leveraging pre-trained policies on layer perception (MLP) based structure, which is not capa-
bleofcapturingsufficientinformationforstate-actionalign-
sourcetaskstoacceleratelearninginatargettask(Rusuetal.
ment.Thismayevenhinderthetransferperformanceontar-
2016;Schmittetal.2018;Parisotto,Ba,andSalakhutdinov
|     |     |     |     |     |     |     | get tasks | with large | state-action |     | spaces. | Please | refer | to the |
| --- | --- | --- | --- | --- | --- | --- | --------- | ---------- | ------------ | --- | ------- | ------ | ----- | ------ |
2016;Yangetal.2020a,b;Taoetal.2021).However,these
appendix1formoreinformationonrelatedwork.
| methods | assume source | tasks | share | the same | state-action |     |            |       |             |     |            |     |         |        |
| ------- | ------------- | ----- | ----- | -------- | ------------ | --- | ---------- | ----- | ----------- | --- | ---------- | --- | ------- | ------ |
|         |               |       |       |          |              |     | To address | these | challenges, |     | we propose |     | a novel | trans- |
*Correspondingauthor ferapproachcalledTURRET(TransferUsinggRaphneuRal
Copyright©2024,AssociationfortheAdvancementofArtificial
Intelligence(www.aaai.org).Allrightsreserved. 1https://github.com/tianpeiyang/TURRET code
16352

TheThirty-EighthAAAIConferenceonArtificialIntelligence(AAAI-24)
|     |     |     |     |     |     |     | countedreturnR= |     | PT  | γi−tr | .   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | --- | ----- | --- | --- | --- |
|     |     |     |     |     |     |     |                 |     | i=t |       | i   |     |     |
PolicyGradientAlgorithmsPolicygradientmethodsare
widelyusedtodirectlyoptimizethepolicyπparameterized
|     |     |     |     |     |     |     | by θ. One | of the | most effective |     | policy gradient | methods | is  |
| --- | --- | --- | --- | --- | --- | --- | --------- | ------ | -------------- | --- | --------------- | ------- | --- |
ProximalPolicyOptimization(PPO)(Schulmanetal.2017),
whichcanavoidthelargedeviationoftheresultscausedby
theuseofimportancesampling.PPOattemptstolearnanew
|     |     |     |     |     |     |     | policyπ θ         | andmakesurethatthedifferencebetweenπ |                                 |     |     |     | θ and |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | ------------------------------------ | ------------------------------- | --- | --- | --- | ----- |
|     |     |     |     |     |     |     | therolloutpolicyπ |                                      | issmall,whichisachievedbyintro- |     |     |     |       |
θold
ducingaconstraint:
|     |     |     |     |     |     |     |        | h   | (cid:16) |         |                |     | (cid:17)i |
| --- | --- | --- | --- | --- | --- | --- | ------ | --- | -------- | ------- | -------------- | --- | --------- |
|     |     |     |     |     |     |     | Lθ =−E | min | r (θ)Aˆ  | ,clip(r | (θ),1−ε,1+ε)Aˆ |     |           |
|     |     |     |     |     |     |     | PPO    | τ   | t        | t       | t              |     | t         |
πθ(at|st)
|     |     |     |     |     |     |     | wherer t (θ)= |     |     | istheratiooftheactionprobabil- |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | --- | ------------------------------ | --- | --- | --- |
πθold (at|st)
Figure1:AmotivatingexampleofTURRET ities under the rollout policy and current policy, and Aˆ is
t
|     |     |     |     |     |     |     | the estimated | advantage. |     | The value | network | V is | updated |
| --- | --- | --- | --- | --- | --- | --- | ------------- | ---------- | --- | --------- | ------- | ---- | ------- |
ψ
withtemporaldifferencelearning:Lψ
|     |     |     |     |     |     |     |     |     |     |     |       | = −E [(V | (s )− |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | -------- | ----- |
|     |     |     |     |     |     |     |     |     |     |     | P P O | τ        | ψ t   |
nETworks). Figure 1 shows a motivating example of trans- Vtarg)2].TheoverallPPOminimizati
|                  |        |           |        |      |           |       | t   |     |     |     | o n objectiv | eis: |     |
| ---------------- | ------ | --------- | ------ | ---- | --------- | ----- | --- | --- | --- | --- | ------------ | ---- | --- |
| ferring policies | across | different | robots | with | different | sizes |     |     |     |     |              |      |     |
and morphologies, where TURRET learns a unified embed- L (θ,ψ)=Lθ +Lψ
|            |         |           |             |        |     |          |     |     | PPO |     | PPO | PPO |     |
| ---------- | ------- | --------- | ----------- | ------ | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
| ding space | for all | the tasks | using Graph | Neural |     | Networks |     |     |     |     |     |     |     |
(GNNs)andthenadaptivelyacceleratesthelearningprocess Cross-Domain Transfer In cross-domain transfer,
|     |     |     |     |     |     |     | the state-action |     | spaces | of  | different | tasks are | differ- |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ------ | --- | --------- | --------- | ------- |
ofthetargettaskwithlargestate-actionspacesorcompletely
|     |     |     |     |     |     |     | ent, i.e., | M   | = ⟨S | ,A ,R | ,T ,γ | ⟩ and | M = |
| --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ---- | ----- | ----- | ----- | --- |
differentmorphologiesbytransferringknowledgefrommul- S S S S S S T
|     |     |     |     |     |     |     | ⟨S ,A ,R | ,T  | ,γ ⟩.Formally,theproblemofmulti-source |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------- | --- | -------------------------------------- | --- | --- | --- | --- |
tiple source policies. The key insights of this paper are the T T T T T
two mechanisms proposed: a Structured Policy Network, cross-domain TL is defined as follows: it includes a series
which can improve the training and transfer performance of source MDPs Π M = {M 1 ,M 2 ,··· ,M n }, where M i
by taking into account the morphological information, and representsthei-thsourceMDP,andatargetMDPM .The
T
|            |        |           |           |           |     |          | MDPs share | some | high-level    |         | commonalities | (e.g., | reptile    |
| ---------- | ------ | --------- | --------- | --------- | --- | -------- | ---------- | ---- | ------------- | ------- | ------------- | ------ | ---------- |
| a adaptive | policy | transfer, | which can | determine |     | when and |            |      |               |         |               |        |            |
|            |        |           |           |           |     |          | robots may | have | qualitatively | similar | gaits).       | The    | goal is to |
whichsourcepoliciesshouldbetransferredtothetargettask.
|             |                   |     |     |             |     |        | accelerate | the learning | process |     | on the target | task | by selec- |
| ----------- | ----------------- | --- | --- | ----------- | --- | ------ | ---------- | ------------ | ------- | --- | ------------- | ---- | --------- |
| In summary, | our contributions |     | are | as follows: | 1)  | TURRET |            |              |         |     |               |      |           |
adoptsanattentionmechanisminthestructuredpolicynet- tivelytransferringthemostbeneficialknowledgefromΠ M .
worktocapturedifferentrelationshipsofneighboringnodes Transferability Measurement Some works explicitly
|            |               |        |                 |     |             |     | calculate | the similarity      |     | between | MDPs,     | which is | used to |
| ---------- | ------------- | ------ | --------------- | --- | ----------- | --- | --------- | ------------------- | --- | ------- | --------- | -------- | ------- |
| to learn a | more semantic | node   | representation, |     | maintaining |     |           |                     |     |         |           |          |         |
|            |               |        |                 |     |             |     | measure   | the transferability |     | between | different | tasks    | (Hu,    |
| sufficient | information   | during | the aggregation |     | process     | and |           |                     |     |         |           |          |         |
Gao,andAn2015b,a).However,thesemethodsusuallyre-
leadingtoacommonstateembeddingspaceforalltasks.2)
measures the distance of states in the unified em- quireaknownworldmodelandhaveahighcomputational
TURRET
bedding space to measure the similarity at each state from complexity,whichcannotbeappliedtomorecomplexcon-
tinuouscontroltasks.Incontrast,otherworksimplicitlyuse
| multiple | cross-domain | source | policies. | In  | this way, | TUR- |             |             |     |        |             |            |     |
| -------- | ------------ | ------ | --------- | --- | --------- | ---- | ----------- | ----------- | --- | ------ | ----------- | ---------- | --- |
|          |              |        |           |     |           |      | the average | performance |     | on the | target task | to measure | the |
RETachievesadaptiveanddelicatetransfer.3)TURRETcan
transferabilityofeachsourcepolicy(Ferna´ndezandVeloso
beeasilycombinedwithexistingDRLalgorithms.sinceno
additional optimization objectives are required in the train- 2006; Li and Zhang 2018; You et al. 2022). However, us-
ingprocess.ExperimentalresultsshowthatTURRETsignif- ingaverageperformancecannothandlethesituationwhere
onlyapartoftheinformationindifferentsourcepoliciesis
icantlyoutperformsthestate-of-the-artmethodsoncontinu-
|     |     |     |     |     |     |     | useful. Previous |     | works | have supported |     | this point | that each |
| --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | ----- | -------------- | --- | ---------- | --------- |
ouscontroltasks.
sourcetaskshouldbemorebeneficialinacertainpartofthe
Preliminaries state space (Yang et al. 2020a) or a single state (Rajendran
etal.2017).Nevertheless,alltheaboveworksarestillcon-
Reinforcement Learning RL problems are typically for- strainedbytheassumptionofthesamestate-actionspace.
malized as Markov decision processes (MDPs). An MDP Graph Neural Networks (GNNs) Traditional methods
can be described as a tuple M = ⟨S,A,R,T,γ⟩, where that use MLP as the policy model cannot handle the cross-
S and A are the sets of states and actions, respectively; domaintransferproblembecauseitcanonlyaccepttheinput
T : S ×A×S 7→ [0,1] is the transition probability dis- andoutputwiththesamedimensionacrosstasks.Incontrast,
R
tribution over states; R : S ×A×S 7→ is the reward undertheconstraintsofdifferentstate-actionspaces,GNNs
function which gives returns on the agent’s performance; havebeenanaturalchoiceformodelingpoliciesduetotheir
and γ is the discount factor for future rewards. A policy capacitytohandlegraphsofvaryingsizes.
π : S ×A 7→ [0,1] is defined as a state-conditioned prob- A graph is denoted as a tuple G = (V,E), where V is
abilitydistributionoveractionsandthegoaloftheagentis a set of nodes and edges E = {(u,v)|u,v ∈ V}. Each
to find an optimal policy π∗ maximizing the expected dis- node and edge have the corresponding representation in a
16353

TheThirty-EighthAAAIConferenceonArtificialIntelligence(AAAI-24)
labeled graph, and a GNN is a function that takes a la- efficientknowledgetransfernotonlyacrossrobotswithhigh
beledgraphasinputandoutputsagraphwithnewlabelsbut similarities in the number of joints and morphological in-
shares the same topology. A general framework of GNNs formation but also across robots with significantly distinct
is the message-passing neural network (MPNN) (Gilmer structuresornumbersofjoints.
et al. 2017), which treats the aggregation process as a K- Structured Policy Network As shown in the previous
step message-passing process. An MPNN framework con- section,previousGNN-basedmethodsareincapableofdis-
sistsofthemessagefunctionsMk andupdatefunctionsUk, tinguishingdifferentcontributionsofneighbornodestothe
whichareusedtocomputethehiddenstatehk+1
|      |     |     |     |     |     |     | andmes- | centralnode,resultinginconsiderableinformationlossdur- |     |     |     |     |     |     |     |
| ---- | --- | --- | --- | --- | --- | --- | ------- | ------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- |
| mk+1 |     |     |     |     |     | v   |         |                                                        |     |     |     |     |     |     |     |
sage for each node v, where k ∈ {0,...,K − 1}. ing the aggregation process, and undesired performance in
v
At each step k, the message function aggregates the hid- tasks with large state-action spaces. The key insight of the
denstatesandedgefeaturesetocomputemessages,which structurepolicynetworkistoalleviatetheinformationloss
arethenpassedintotheupdatefunctiontocomputehidden problemduringtheaggregationprocess.Tocombatthis,we
statesforthenextstep. adoptanattentionmechanismtolearnamoresemanticnode
NerveNet,Snowflake,andCATTheabilityofGNNsto representation, which is described in the following section.
process graphs of arbitrary sizes has been successfully ap- Then, we concatenate node representations and feed them
pliedtoRLincontinuouscontroltasks.Onenotableworkis intoareadoutnetworkF ,thusleadingtoacommonstate
read
NerveNet(Wangetal.2018),whichmodelsthemorphology embeddingspace(seeFigure2(a)).Thesubsequenttransfer
oftherobotasagraphandusesaGNNfollowingthebasic
mechanismisdescribedconsequently.
frameworkofMPNNasthepolicynetwork.Forexample,a AdaptivePolicyTransferThekeyinsightofthispartis
Centipede agent has four different node types: root, torso, that the transferability of each source policy could be re-
hip, and ankle. The torso nodes share the same instance of flectedbythedistanceofembeddingsintheunifiedembed-
the input function, and each torso sends the same message dingspace,whichisobtainedbyourproposedGNN-based
| to the righthip |     | and lefthip. | However, |     | in order | to generalize |     |            |     |                 |     |       |              |          |     |
| --------------- | --- | ------------ | -------- | --- | -------- | ------------- | --- | ---------- | --- | --------------- | --- | ----- | ------------ | -------- | --- |
|                 |     |              |          |     |          |               |     | structured |     | policy network. |     | Using | the obtained | distance | to  |
to different agents with varied node and edge types, Ner- calculatetheweightingfactorsofeachsourcepolicyateach
veNetmergesallothernodesexceptrootasjointinthespe- state(Figure2(b)),i.e.,thesimilaritymetric,wecanadap-
cificimplementation.Differentnodesareusuallycorrelated tivelyanddelicatelyextractknowledgefrommultiplesource
withvaryingdegrees,butduetothisoperationandthechar- policies,whichisdescribedindetailconsequently.
| acteristics | of MPNN, |     | NerveNet | loses | the ability | to  | capture |     |     |     |     |     |     |     |     |
| ----------- | -------- | --- | -------- | ----- | ----------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
thiscorrelation,resultingininformationlossduringtheag- StructuredPolicyNetwork
gregationprocess.Furthermore,asthenumberofnodesin-
|                   |                 |     |       |             |         |               |        | In  | this section, | to        | handle  | the mismatch |            | of state | spaces  |
| ----------------- | --------------- | --- | ----- | ----------- | ------- | ------------- | ------ | --- | ------------- | --------- | ------- | ------------ | ---------- | -------- | ------- |
| creases,          | this phenomenon |     | is    | exacerbated | due     | to the        | multi- |     |               |           |         |              |            |          |         |
|                   |                 |     |       |             |         |               |        | of  | all the       | tasks, we | propose | a            | structured | policy   | network |
| hop communication |                 | in  | GNNs, | thus        | leading | to diminished |        |     |               |           |         |              |            |          |         |
basedonGNNs.Further,weassigndifferentweightingfac-
performanceontaskswithlargestate-actionspaces.
|     |     |     |     |     |     |     |     | tors | to neighboring |     | nodes | to reduce | information |     | loss dur- |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | -------------- | --- | ----- | --------- | ----------- | --- | --------- |
Recently, Blake et al. (2021) regard the above phe- ing the aggregation process. Given a set of N source tasks
| nomenon | as caused | by  | the overfitting |     | problem. | Therefore, |     |      |           |                                               |     |     |     |     |     |
| ------- | --------- | --- | --------------- | --- | -------- | ---------- | --- | ---- | --------- | --------------------------------------------- | --- | --- | --- | --- | --- |
|         |           |     |                 |     |          |            |     | {χ 1 | ,χ 2 ,··· | ,χ n }thathavearbitrarystate-actionspaces,the |     |     |     |     |     |
Snowflakefreezesnetworkparametersduringtraining,thus
structuredpolicynetworkisusedtolearnthepolicyoneach
| facilitating | training | GNNs | on  | larger | graphs | for locomotion |     |     |     |     |     |     |     |     |     |
| ------------ | -------- | ---- | --- | ------ | ------ | -------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
sourcetask,whichcontainsfourmainmodels:inputmodel
| control in | RL. | Unfortunately, |     | Snowflake | requires |     | extra hu- |     |               |       |     |         |       |         |        |
| ---------- | --- | -------------- | --- | --------- | -------- | --- | --------- | --- | ------------- | ----- | --- | ------- | ----- | ------- | ------ |
|            |     |                |     |           |          |     |           | F   | , propagation | model | P,  | readout | model | F , and | output |
|            |     |                |     |           |          |     |           | in  |               |       |     |         |       | read    |        |
man effort to determine which parameters to freeze, which modelF .Wecollectivelyrefertothefirstthreemodelsas
out
wouldbeinfeasibleincomplicatedtasks.
representationmodelG.
| More                                              | recently,        | You | et al.       | (2022) | firstly  | propose | CAT      |              |     |              |                                     |          |              |     |      |
| ------------------------------------------------- | ---------------- | --- | ------------ | ------ | -------- | ------- | -------- | ------------ | --- | ------------ | ----------------------------------- | -------- | ------------ | --- | ---- |
|                                                   |                  |     |              |        |          |         |          | InputModel:s |     |              | istheagentobservationvectorobtained |          |              |     |      |
| to solve                                          | the multi-source |     | cross-domain |        | transfer |         | problem, |              |     | t            |                                     |          |              |     |      |
|                                                   |                  |     |              |        |          |         |          | from         | the | environment, | which                               | contains | observations |     | x of |
| whichisthesamesettingasinthispaper.CATlearnsaone- |                  |     |              |        |          |         |          |              |     |              |                                     |          |              |     | v    |
eachnodevcorrespondingtoeachjoint(Wangetal.2018).
to-onemappingbetweeneachsourcetaskandthetargettask
|     |     |     |     |     |     |     |     | Then,theinitialnoderepresentationh0 |     |     |     |     | atpropagationstep |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------------------------- | --- | --- | --- | --- | ----------------- | --- | --- |
v
toextractusefulknowledgefrommultiplesourcepolicynet-
|                                                     |     |     |     |     |     |     |     | 0 is    | obtained | by placing | the      | node                | vector into | an  | input net- |
| --------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ------- | -------- | ---------- | -------- | ------------------- | ----------- | --- | ---------- |
| works.However,CATlacksgeneralizationtomorerealistic |     |     |     |     |     |     |     | work:h0 |          |            |          |                     |             |     |            |
|                                                     |     |     |     |     |     |     |     |         |          | = F (x v   | ),whereF | isanMLPandwekeepthe |             |     |            |
| scenarioswhenthenumberofsourcetaskschanges.More-    |     |     |     |     |     |     |     |         | v        | in         |          | in                  |             |     |            |
fixed-sizeinputforothernodeobservationsofdifferentsizes
over,thetransferabilitymeasurementusedinCATisnotdel-
bypaddingzerostothevectors.
icateenoughtohandlethesituationwhereeachsourcepol-
|     |     |     |     |     |     |     |     | Propagation |     | Model: | As  | described | before, | the | informa- |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ------ | --- | --------- | ------- | --- | -------- |
icyperformsbetterinonlyapartofthestatespace.
tionlossphenomenonduringtheaggregationprocessresults
|     |     |     |     |     |     |     |     | in diminished |     | performance |     | on tasks | with | large state-action |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ----------- | --- | -------- | ---- | ------------------ | --- |
Methodology
spaces,whichwe’llalsoshowintheempiricalsection.This
ThissectionfirstintroducesthewholestructureofTURRET phenomenonwillbealleviatedbyconsideringthedifferent
andthendescribeseachcomponentofTURRETindetail. contributionsofdifferentneighbornodestothecentralnode.
Tocapturedifferentrelativeweightsbetweennodes,wein-
FrameworkOverview
troducetheattentionmechanism(Velicˇkovic´etal.2018)(we
Figure2illustratestheoverallframeworkofTURRET,which
usemulti-headattentioninpractice):
| contains | two main     | components: |        | (a)       | structured | policy | net-        |     |      |                                  |     |     |     |     |     |
| -------- | ------------ | ----------- | ------ | --------- | ---------- | ------ | ----------- | --- | ---- | -------------------------------- | --- | --- | --- | --- | --- |
|          |              |             |        |           |            |        |             |     | αk+1 | =Softmax(ð(aT[Wk+1hk||Wk+1hk])), |     |     |     |     |     |
| work and | (b) adaptive |             | policy | transfer. | TURRET     |        | facilitates |     |      |                                  |     |     |     |     |     |
|          |              |             |        |           |            |        |             |     | vu   |                                  |     |     | v   |     | u   |
16354

TheThirty-EighthAAAIConferenceonArtificialIntelligence(AAAI-24)
Figure2:TURRETcontainstwomaincomponents.First,thestructuredpolicynetworkarchitectureisshowninthebrownbox,
which handles the mismatch of state spaces of all the tasks. Second, the adaptive policy transfer architecture is shown in the
blue call-out box, which adaptively and delicately extracts knowledge from multiple source policies. Each task has separate
F ,P,F ,F ,andthereadoutmodelF mapsthestatestoaunifiedembeddingspace,whichisusedtomeasuredistance
| in  | read out |     |     |     | read |     |     |     |     |     |     |
| --- | -------- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- |
andgenerateweightingfactors.
where ð(·) is the LeakyReLu activation function, Wk a where [·] refers to computation specific to head k. To
k
weight matrix and a a vector of learnable parameters. The thispoint,allthetasksobtaintheirstaterepresentationsina
attentioncoefficientαk
measurestherelationshipbetween commonspace,whichcontainsmorestructuralinformation
vu
the node v and its neighbor node u. Then the node update acrosstasksandadaptstovaryingtypesornumbersoftasks.
functionisasfollows: Output Model: Previous GNN-based methods predict
theactionsofeachnodewiththefinalnoderepresentations
X
hk +1 =σ( α k +1Wk+1hk ). as input, which only supports zero-shot transfer and is not
|     | v   |     |        | v u | u   |     |                                                     |     |     |     |     |
| --- | --- | --- | ------ | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- |
|     |     |     | u∈N(v) |     |     |     | conducivetothedesignofthetransferprocess.Instead,we |     |     |     |     |
takestaterepresentationsintoanoutputnetworkandpredict
Theattentionmechanismcannotonlygeneralizetodifferent
theactiondistributionforallnodes:
| tasks but | also capture |     | the important |     | relationships | between |     |     |         |     |     |
| --------- | ------------ | --- | ------------- | --- | ------------- | ------- | --- | --- | ------- | --- | --- |
|           |              |     |               |     |               |         |     |     | µ =F (S | )   |     |
nodes, which reduces information loss during the aggrega- v∈V out emb
| tion process                                      | and | improves | the | training | and transfer | perfor- |         |        |             |                       |     |
| ------------------------------------------------- | --- | -------- | --- | -------- | ------------ | ------- | ------- | ------ | ----------- | --------------------- | --- |
|                                                   |     |          |     |          |              |         | where µ | is the | mean of the | Gaussian distribution | and |
| mancerelativetootherGNN-basedmethodsonlargedimen- |     |          |     |          |              |         |         | v∈V    |             |                       |     |
thestandarddeviationisatrainablevector.WechoosePPO
sionalcontroltasks.
(Schulmanetal.2017)totrainalllearnableparametersinan
| Readout | Model: | Inspired | by  | adaptive | readout | functions |     |     |     |     |     |
| ------- | ------ | -------- | --- | -------- | ------- | --------- | --- | --- | --- | --- | --- |
end-to-endmanner.
proposedby(Buterezetal.2022),TURRETadoptssettrans-
| former | readouts | to learn | an  | overall | state representation. |     |     |     |     |     |     |
| ------ | -------- | -------- | --- | ------- | --------------------- | --- | --- | --- | --- | --- | --- |
AdaptivePolicyTransfer
| Specifically, | once | the | propagation | model | obtains | the final |              |           |                   |         |          |
| ------------- | ---- | --- | ----------- | ----- | ------- | --------- | ------------ | --------- | ----------------- | ------- | -------- |
|               |      |     |             |       |         |           | This section | describes | how to adaptively | extract | the most |
noderepresentations,thenodevectorsarefirstcollectedinto
RM×D,whereM relevantknowledgefrommultiplesourcestructuredpolicies
| amatrixH | ∈   |     |     | isthemaximalnumberof |     |     |     |     |     |     |     |
| -------- | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- | --- |
toacceleratethelearningprocessforcross-domaintransfer.
| nodes in              | source | and target | tasks  | and  | D is the  | dimension of |         |               |            |               |             |
| --------------------- | ------ | ---------- | ------ | ---- | --------- | ------------ | ------- | ------------- | ---------- | ------------- | ----------- |
|                       |        |            |        |      |           |              | Instead | of outputting | the action | of each joint | separately, |
| node representations. |        | For        | graphs | with | less than | M nodes,     |         |               |            |               |             |
wecombinenoderepresentationstolearnaglobalstaterep-
wesetthepaddedvaluestozerotoensurethatcurrentstates
resentationthroughthereadoutmodel,whichcanreflectse-
canbefedintothereadoutmodelsofsourcetasks.Thenthe
manticenvironmentalinformation.Tothisend,thedistance
| matrixH | isfedintothesettransformerreadoutfunctionas |     |     |     |     |     |     |     |     |     |     |
| ------- | ------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ofstatesintheembeddingspacewithsufficientsemanticin-
inputandstateembeddingsaregeneratedusinganattention-
formationcanbeseenasasuitablemetrictoreflectthetrans-
basedencoder-decodermodule(describedinappendix):
|     |     |     |     |     |     |     | ferability    | of each source | policy           | at the current | state. More |
| --- | --- | --- | --- | --- | --- | --- | ------------- | -------------- | ---------------- | -------------- | ----------- |
|     |     |     |     |     |     |     | specifically, | we first       | obtain the state | representation | s of        |
|     |     |     | K   |     |     |     |               |                |                  |                | emb         |
|     |     | 1   | X   |     |     |     |               |                |                  |                |             |
S =F (H)= [DECODER(ENCODER(H))] t h e c u r r e n t s ta te s t t h ro ug h t h e re p re s en t a t io nm o d e l G in t h e
| emb | read |     |     |     |     | k   |     |     |     |     |     |
| --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
K t ar g e t t a s k . A t th e s a m e ti m e , s i s fe d i n t o G of e a c h so u r ce
|     |     |     | k=1 |     |     |     |     |     | t   |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
16355

TheThirty-EighthAAAIConferenceonArtificialIntelligence(AAAI-24)
tasktoobtain{s ,s ,...,s },correspondingto Environments:Wetestourmethodonasetofcontinuous
|     |     | emb1 emb2 |     | embn |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{χ ,χ ,··· ,χ }. The distance between state embeddings controltasksinMuJoCo(Todorov,Erez,andTassa2012).In
| 1 2 |     | n   |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
in the common space is calculated to generate weights of additiontocommonlyusedtaskssuchasWalker,Ant,and
eachsourcepolicyasfollows: Humanoid,wealsouseasetofCentipede-ntasks,eachof
|                 |     |     |     |                       |     |     | which has | a robot | with | n/2 | torso bodies | and | n legs | (Wang |
| --------------- | --- | --- | --- | --------------------- | --- | --- | --------- | ------- | ---- | --- | ------------ | --- | ------ | ----- |
| ω =Softmax((||s |     | −s  |     | ||2)−1),i∈{1,2,...,n} |     |     |           |         |      |     |              |     |        |       |
i emb embi 2 etal.2018).Amoredetaileddescriptionisinappendix.
Forexample,thecurrentstates inanoctopodrobotisfed Baselines:weconsiderthefollowingsixbaselines:
t
intoG ofaquadrupedandG
| 1         |           |                                       | 2 ofahexapodrobot,obtaining |     |     |     |       |           |     |            |     |            |     |        |
| --------- | --------- | ------------------------------------- | --------------------------- | --- | --- | --- | ----- | --------- | --- | ---------- | --- | ---------- | --- | ------ |
|           |           |                                       |                             |     |     |     | • PPO | (Schulman | et  | al. 2017), | a   | mainstream | RL  | method |
| s emb1 ,s | emb2 andω | 1 ,ω 2 (seeFigure2(a)).Throughthecom- |                             |     |     |     |       |           |     |            |     |            |     |        |
thatlearnsfromscratchinthetargettask;
monembeddingspace,wecannotonlyhandlethemismatch
of different state spaces but also achieve adaptive transfer • CAT(Youetal.2022),whichadaptivelyextractsknowl-
ateachstatewithsufficientinterpretability.Withthegener- edgefrommultiplecross-domainsourcepolicies;
| atedweights, |        | adoptsthetransfermethodoflateral |     |     |     |     |                                                  |     |     |     |     |     |     |     |
| ------------ | ------ | -------------------------------- | --- | --- | --- | --- | ------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- |
|              | TURRET |                                  |     |     |     |     | • NerveNet(Wangetal.2018),whichmodelsthemorphol- |     |     |     |     |     |     |     |
connections between the source and target networks (Liu, ogyinformationintoGNNstorepresentthepolicy;
Peng,andSchwing2019;Wan,Gangwani,andPeng2020;
|            |        |               |     |        |                    |     | • Snowflake | (Blake | et  | al. 2021), | which | extends | NerveNet |     |
| ---------- | ------ | ------------- | --- | ------ | ------------------ | --- | ----------- | ------ | --- | ---------- | ----- | ------- | -------- | --- |
| You et al. | 2022). | Specifically, | we  | denote | the pre-activation |     |             |        |     |            |       |         |          |     |
tohigh-dimensionalcontinuouscontrolenvironments;
| outputs | of the | j-th hidden | layers | of the | i-th source | policy |     |     |     |     |     |     |     |     |
| ------- | ------ | ----------- | ------ | ------ | ----------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
{zj,1 • NerveNet/Snowflake+fine-tune, which directly uses the
| network | as  | ≤ i | ≤ N,1 | ≤ j ≤ | N π }. | The policy |             |     |         |           |        |     |                    |     |
| ------- | --- | --- | ----- | ----- | ------ | ---------- | ----------- | --- | ------- | --------- | ------ | --- | ------------------ | --- |
|         |     | i   |       |       |        |            | old weights |     | trained | on source | models |     | for initialization |     |
andvaluenetworkshavethesamenumberofhiddenlayers
andthencontinuestotraininthetargettask.
N inoursettingandweonlydiscusspolicynetworkshere.
π
Then, combines the representations zj in the tar- • SWAT (Hong, Yoon, and Kim 2022), which adopts a
TURRET
transformerstructureinmulti-tasktraining.
| get network                | with | those from | source | networks | at  | each state |              |     |     |     |     |     |     |     |
| -------------------------- | ---- | ---------- | ------ | -------- | --- | ---------- | ------------ | --- | --- | --- | --- | --- | --- | --- |
| followingweightingfactorsω |      |            | i as:  |          |     |            |              |     |     |     |     |     |     |     |
|                            |      |            |        | N        |     |            | SizeTransfer |     |     |     |     |     |     |     |
X
|     |     | zj =pzj | +(1−p) | ω   | zj  |     |                                                      |     |     |     |     |     |     |     |
| --- | --- | ------- | ------ | --- | --- | --- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
|     |     | π       |        |     | i i |     | Insizetransfer,weconsiderCentipede-4andCentipede-6as |     |     |     |     |     |     |     |
|     |     |         |        | i=1 |     |     | oursourcetasks.Forthetargettasks,wechooseCentipede-  |     |     |     |     |     |     |     |
where p isan increasingfactor overtimeto controlthe de- {12,16,20},whichareverydifficulttotrainfromscratch.
gree of independence of the target network (detailed in ap- Figure 3 (a)-(c) depicts the performance of and
TURRET
|     |     |     |     |     |     |     | other baseline |     | methods | across | three | experimental |     | scenar- |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ------- | ------ | ----- | ------------ | --- | ------- |
pendix).Atthebeginningofthetrainingprocess,theagent
needs more assistance from source policies. As the agent ios. NerveNet exhibits inferior performance due to grap-
gainsmoreknowledge,itreliesmoreonitself,whichiscon- pling with the intricacies of dealing with a large number
trolled by a higher value of p. In this way, TURRET adap- of joints. While CAT demonstrates competitive results in
tivelyextractsknowledgefrommultiplesourcepoliciesand the Centipede-12 task, it progressively falls behind TUR-
RETasthenumberofjointsincreases.Thistrendhighlights
avoidsnegativetransfer,achievingmoreefficienttransfer.
|     |     |     |     |     |     |     | the limitations |     | of MLP-based |     | policies | in addressing |     | struc- |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ------------ | --- | -------- | ------------- | --- | ------ |
Experiments turaldisparitiesinrobotswithvaryingjointcounts.Further-
more,NerveNet/Snowflake+fine-tunehasaslightjumpstart
| In this section, |     | we present | four | types of | transfer | learning |              |             |     |             |     |          |     |          |
| ---------------- | --- | ---------- | ---- | -------- | -------- | -------- | ------------ | ----------- | --- | ----------- | --- | -------- | --- | -------- |
|                  |     |            |      |          |          |          | and marginal | performance |     | improvement |     | compared |     | to their |
experimentsthatcoverawiderangeofenvironmentstover-
|                       |     |     |     |              |               |     | vanilla | forms. Significantly, |     |     | prior | GNN-based | approaches |     |
| --------------------- | --- | --- | --- | ------------ | ------------- | --- | ------- | --------------------- | --- | --- | ----- | --------- | ---------- | --- |
| ify the effectiveness |     | of  |     | from various | perspectives. |     |         |                       |     |     |       |           |            |     |
TURRET
Thefirsttypeisleveragingasetofsmallsourcetaskmodels falter in transferring knowledge to agents with particularly
|     |     |     |     |     |     |     | large differences |     | in the | number | of  | joints, which | is  | exactly |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | ------ | ------ | --- | ------------- | --- | ------- |
toacceleratealargertargettask,i.e.,sizetransfer.Thesec-
whatourapproachseekstoovercome.Similarly,SWAT’sin-
ondtypeconsiderssourceandtargettaskswhererobotshave
efficientsimultaneousaggregationofjointinformationham-
| entirely      | different | controls | and constructions, |                | i.e., | morphol- |          |              |     |            |     |        |         |          |
| ------------- | --------- | -------- | ------------------ | -------------- | ----- | -------- | -------- | ------------ | --- | ---------- | --- | ------ | ------- | -------- |
|               |           |          |                    |                |       |          | pers its | performance, |     | compounded |     | by its | lack of | adaptive |
| ogy transfer. | These     | first    | two types          | of experiments |       | evaluate |          |              |     |            |     |        |         |          |
theeffectivenessofmulti-sourcecross-domainTLmethods. transfercapability.Conversely,TURRETconsistentlyoutper-
|           |      |                 |              |            |        |             | forms all     | baselines      | across   | all       | test      | environments.  |           | This can |
| --------- | ---- | --------------- | ------------ | ---------- | ------ | ----------- | ------------- | -------------- | -------- | --------- | --------- | -------------- | --------- | -------- |
| The third | type | converts        | trajectories | in the     | source | and tar-    |               |                |          |           |           |                |           |          |
|           |      |                 |              |            |        |             | be attributed | to             | TURRET’s | adeptness |           | at considering |           | diverse  |
| get tasks | into | a 3-dimensional | space        | to perform |        | qualitative |               |                |          |           |           |                |           |          |
|           |      |                 |              |            |        |             | neighbor      | contributions, |          | thus      | capturing | node           | semantics | and      |
andquantitativeanalysis,whichshowshowTURRETworks.
beingabletomeasurestatedistancesinaunifiedembedding
Fourth,wepresentasetofablationstudiestoevaluatediffer-
entcomponentsofTURRET.Moreexperimentsonthestruc- space, enabling adaptive knowledge extraction from multi-
turedpolicynetworktrainingonlargeagentsandthesetting plesourcepolicies,whichisalsovalidatedinthefollowing
experiments.
ofmorethantwosourcetaskscanbefoundinappendix.For
convenienceandcomputationalcomplexity,wetypicallyset
MorphologyTransfer
| thenumberofsourcetasksN |     |     | to2.Werun5randomseeds |     |     |     |     |     |     |     |     |     |     |     |
| ----------------------- | --- | --- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
foreachalgorithminanexperiment,andeachseedrunsfor Inmorphologytransfer,weconsiderthreecombinationsof
10 million environment interactions (i.e., timesteps). More TL experiments: {Hopper & Centipede-4 → Walker2d},
details of network structures and parameter settings can be {HalfCheetah&Ant→Centipede-8}and{HalfCheetah&
| foundinappendix. |     |     |     |     |     |     | Ant→Humanoid}. |     |     |     |     |     |     |     |
| ---------------- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- |
16356

TheThirty-EighthAAAIConferenceonArtificialIntelligence(AAAI-24)
TURRET CAT SWAT Snowflake Snowflake+fine-tune NerveNet NerveNet+fine-tune PPO
(a) Centipede {4,6} Centipede-12 (b) Centipede {4,6} Centipede-16 (c) Centipede {4,6} Centipede-20
(d) {Hopper, Centipede-4} Walker2d (e) {Halfcheetah, Ant} Centipede-8 (f) {Halfcheetah, Ant} Humanoid
Figure3:PerformanceofTURRETandotherbaselinesondifferentsizesandmorphologiesofcontinuouscontroltasks.Weplot
thenumberoftimestepsofenvironmentinteractiononthex-axisandtheaverageepisodicreturnsonthey-axis(thecurvesand
shadowareasrepresentthemeanandstandarddeviationover5trials,respectively).
WeshowthetrainingcurvesofallthemethodsinFigure3 whichmaymaketransferdifficult.Incontrast,Figures4(b)
(d)-(f). As the results show, CAT achieves comparable per- and(d)showthedifferencebetweensourceandtargetpoli-
formance to TURRET in Centipede-8, because both source cieslearnedusing TURRET,wherethereisahigheroverlap
taskshaverelativelysimilarmorphologiestothetargettask anddistancereduction,relativetothePPOtrajectories.This
and can provide enough knowledge to accelerate the target phenomenonindicatesthatTURRETnarrowsthedistancebe-
tasklearning.However,CATperformsworsethan TURRET tweenstateswithsimilarsemanticsintheembeddingspace,
in the other two tasks because the two source tasks differ thusfacilitatingeffectivetransfer.Similarly,thetrajectories
widely from the target task thus difficult to capture com- inthemorphologytransfersetting(i.e.,Figures4(e)and(g)
monalitiesusingMLP-basedmethods.Finally,TURRETsig- vs. (f) and (h)) also show that our structured source policy
nificantlyimproveslearningefficiencycomparedtoallbase- covers more of the target task policy than the MLP-based
linesinHumanoid,whichhaslargestate-actionspacesandis source policy. We can see that the trajectories of our struc-
hardtolearnfromscratch.AlltheresultsindicatethatTUR- tured policies are more similar in the project space in all
RET facilitates effective cross-domain transfer across tasks examples,indicatingthatTURRETproducesstructuredpoli-
with completely different morphologies, matching or out- cies that capture semantic commonalities of two tasks and
performingallbaselinestested. learnsrepresentationsmoreusefulthanPPO.
VisualAnalysis AblationStudies
Inthissection,weanalyzethecontributionofdifferentparts
This section considers a post-hoc analysis to better under-
standwhy TURRET facilitateseffectivecross-domaintrans- of TURRET to better verify the effectiveness of our trans-
fer. We first collect trajectories from policies learned with
fermethod.WeremovetheattentionmechanisminTURRET
to verify it is useful to aggregate information by adopting
PPOandTURRETindifferenttasks.Second,weprojectthe
differentweightingfactorstoneighbornodeswithoutinfor-
sourceandtargettrajectoriesintoa3-dimensionalspaceus-
mationlossontaskswithlargestate-actionspaces.Wealso
ingt-SNE.Third,wecalculatetheEuclideandistanceDbe-
tweenthecorrespondingstatesoftwotrajectories,whichcan replace our similarity metric in TURRET with the average
performance of each source policy on the target task as a
reflectthesimilaritybetweentwotasks,i.e.,alowervalueof
weighting measurement, which is used to verify the effec-
Dmaymeanamorehelpfultransferastheprojectedsource
tivenessofouradaptivetransfermethod.Theablationstud-
andtargetpoliciesaremoresimilar.
iesaredesignedasfollowsunderexperimentsonCentipede-
Figure4visualizesthisanalysisofsizetransferandmor-
{4,6}→16:
phology transfer. From Figures 4 (a) and (c), we can see
that the source and target policies learned by MLP-based • TURRET w/o attention: Using MPNN without the atten-
PPO differ significantly when projected into 3 dimensions, tionmechanismduringtheaggregationprocess.
16357

TheThirty-EighthAAAIConferenceonArtificialIntelligence(AAAI-24)
(a) 4-16(MLP):33.58 (b) 4-16(TURRET):20.51 (c) 4-20(MLP):54.69 (d) 4-20(TURRET):20.62
(e) H-W(MLP):18.46 (f) H-W(TURRET):16.98 (g) H-H(MLP):21.17 (h) H-H(TURRET):18.96
Figure4:OurvisualanalysisresultsonMuJoCo:Centipede-4→Centipede-{16,20},Hopper→Walker2d,andHalfCheetah→
Humanoid.Theblueandreddotsrepresenttheoptimaltrajectoriesonthesourceandtargettasks,respectively.Forexample,“H-
H(TURRET):18.96”representsthedistributionoftrajectoriessampledbyourstructuredpolicyinthesourcetaskHalfCheetah
andthatinthetargettaskHumanoid,andtheaverageEuclideandistanceDbetweentwotrajectoriesis18.96.
|     |     |     |     |     |     |     | ated by incorporating |     | additional | techniques |     | in TURRET | in  |
| --- | --- | --- | --- | --- | --- | --- | --------------------- | --- | ---------- | ---------- | --- | --------- | --- |
sdraweR edosipE 6000
|     |     |     |     |     |     |     | future work. | First, | this paper | assumes | that | different | tasks |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------ | ---------- | ------- | ---- | --------- | ----- |
5000 have a similar goal, and thus task similarity could be re-
flectedinstatesimilarity.Insomecases,differenttaskshave
4000
differentgoals,e.g.,openingthedoororpushingthebutton.
3000
|      |     |     |     |     |     |     | In this situation,                                | TURRET |     | could integrate |     | the state | similar- |
| ---- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | ------ | --- | --------------- | --- | --------- | -------- |
| 2000 |     |     |     |     |     |     | ityandtheperformanceofeachsourcepolicytodetermine |        |     |                 |     |           |          |
TURRET
whichsourcepolicyismoresuitable.Second,thispaperas-
| 1000 |     |     |     | TURRET w/o attention |     |     |         |            |       |              |       |      |         |
| ---- | --- | --- | --- | -------------------- | --- | --- | ------- | ---------- | ----- | ------------ | ----- | ---- | ------- |
|      |     |     |     |                      |     |     | sumes a | very clean | state | composition, | where | each | feature |
TURRET w/o distance
0 represents the joint of a robot. In practice, states may con-
|     | 0.0 | 0.2 | 0.4 | 0.6 | 0.8 | 1.0  |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
tainnoisyorirrelevantinformation.Representationlearning
 ×107
Training Steps could be combined with TURRET to discard irrelevant fea-
turesviauncertaintymeasurementoradversariallearningto
Figure5:AblationstudiesinCentipede-{4,6}→16 onthe solvethisproblem.
contributionofattentionmechanismandsimilaritymetric.
ConclusionandFutureWork
Inthiswork,weproposeTURRET,aGNN-basedframework
• TURRETw/odistance:Usingtheaverageperformanceas
toachieveadaptivemulti-sourcecross-domainTL.TURRET
weightingfactorstoextractknowledge.
|     |     |     |     |     |     |     | contains | two main | components: | a   | meticulously |     | structured |
| --- | --- | --- | --- | --- | --- | --- | -------- | -------- | ----------- | --- | ------------ | --- | ---------- |
Figure5showstheresultsofourablationstudies.Aswe
policynetworkandadaptiveknowledgetransfer.Byadopt-
| can  | see, TURRET | has | a significant | performance    |       | improve- |                   |            |     |            |     |           |          |
| ---- | ----------- | --- | ------------- | -------------- | ----- | -------- | ----------------- | ---------- | --- | ---------- | --- | --------- | -------- |
|      |             |     |               |                |       |          | ing the attention | mechanism, |     | we capture |     | different | relative |
| ment | compared    | to  |               | w/o attention, | which | confirms |                   |            |     |            |     |           |          |
TURRET weights between nodes, enabling the acquisition of a uni-
theeffectivenessofourattentionmechanism.Besides,TUR-
fiedstateembeddingspacethroughsettransformerreadouts.
RET w/o distance performs worse than TURRET. This in- Basedonthesimilaritymetric, canadaptivelyand
TURRET
| dicates | that | using the | average | performance | as a | weighting |     |     |     |     |     |     |     |
| ------- | ---- | --------- | ------- | ----------- | ---- | --------- | --- | --- | --- | --- | --- | --- | --- |
delicatelyextractknowledgeatthestatelevel.Wealsoper-
| factor | cannot | handle | the situation | where | only a | part of the |               |          |     |            |               |     |            |
| ------ | ------ | ------ | ------------- | ----- | ------ | ----------- | ------------- | -------- | --- | ---------- | ------------- | --- | ---------- |
|        |        |        |               |       |        |             | form a visual | analysis | to  | verify the | effectiveness |     | of captur- |
informationindifferentsourcepoliciesisuseful.Therefore,
ingmorphologicalcommonalitiesacrosstasks.Inthispaper,
| a delicate | transferability |     | metric | should be | facilitated | to im- |          |               |           |     |          |        |      |
| ---------- | --------------- | --- | ------ | --------- | ----------- | ------ | -------- | ------------- | --------- | --- | -------- | ------ | ---- |
|            |                 |     |        |           |             |        | we adopt | the attention | mechanism |     | in GNNs. | Future | work |
provethetransferperformance.Alltheaboveresultsconfirm couldconsiderdesigningaspecificmessage-passingmech-
that TURRET achieves more efficient transfer than previous anismthatcanfullyreflecttherobot’smorphologyinforma-
cross-domaintransfermethods.
tion.AnotherdirectionistoapplyTURRETtomorecompli-
|     |     |     |     |     |     |     | cated scenarios | by  | employing | a more | comprehensive |     | simi- |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | --------- | ------ | ------------- | --- | ----- |
Discussion
laritymeasurementtohandlethegoalmismatchindifferent
This paper focuses on effectively capturing the common- tasks. Furthermore, existing methods always require struc-
alities in different state-action spaces to discover a feature tured information about the robot in advance when dealing
space that can enhance transferability. This pursuit, how- withnodevectors;thus,howtoextractthisinformationau-
ever, reveals two intrinsic limitations, which can be allevi- tomaticallyisworthfurtherstudy.
16358

TheThirty-EighthAAAIConferenceonArtificialIntelligence(AAAI-24)
Acknowledgments Symposium on Educational Advances in Artificial Intelli-
|           |     |           |        |          |         |         | gence (EAAI-18), |     | New | Orleans, | Louisiana, | USA, | February |     |
| --------- | --- | --------- | ------ | -------- | ------- | ------- | ---------------- | --- | --- | -------- | ---------- | ---- | -------- | --- |
| This work | is  | supported | by the | National | Natural | Science |                  |     |     |          |            |      |          |     |
2-7,2018,3562–3570.AAAIPress.
| Foundation | of       | China (Grant | Nos.    | 92370132, |       | 62106172)  |            |        |          |              |     |        |           |     |
| ---------- | -------- | ------------ | ------- | --------- | ----- | ---------- | ---------- | ------ | -------- | ------------ | --- | ------ | --------- | --- |
|            |          |              |         |           |       |            | Lillicrap, | T. P.; | Hunt, J. | J.; Pritzel, | A.; | Heess, | N.; Erez, | T.; |
| and the    | National | Key R&D      | Program | of        | China | (Grant No. |            |        |          |              |     |        |           |     |
Tassa,Y.;Silver,D.;andWierstra,D.2016.Continuouscon-
| 2022ZD0116402). |     | Part | of this work | has | taken | place in the |     |     |     |     |     |     |     |     |
| --------------- | --- | ---- | ------------ | --- | ----- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
Intelligent Robot Learning (IRL) Lab at the University of trolwithdeepreinforcementlearning. InProceedingsofthe
Alberta,whichissupportedinpartbyresearchgrantsfrom 4thInternationalConferenceonLearningRepresentations.
Liu,I.;Peng,J.;andSchwing,A.G.2019.KnowledgeFlow:
theAlbertaMachineIntelligenceInstitute(Amii);aCanada
CIFARAIChair,Amii;ComputeCanada;Huawei;Mitacs; Improve Upon Your Teachers. In Proceedings of the 7th
| andNSERC. |     |     |     |     |     |     | InternationalConferenceonLearningRepresentations. |     |     |     |     |     |     |     |
| --------- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
Mnih,V.;Kavukcuoglu,K.;Silver,D.;Rusu,A.A.;Veness,
References J.;Bellemare,M. G.;Graves,A.;Riedmiller,M.A.; Fidje-
land,A.;Ostrovski,G.;Petersen,S.;Beattie,C.;Sadik,A.;
| Blake, | C.; Kurin, | V.; | Igl, M.; | and | Whiteson, | S. 2021. |     |     |     |     |     |     |     |     |
| ------ | ---------- | --- | -------- | --- | --------- | -------- | --- | --- | --- | --- | --- | --- | --- | --- |
Antonoglou,I.;King,H.;Kumaran,D.;Wierstra,D.;Legg,
| Snowflake: | Scaling | GNNs | to high-dimensional |     |     | continuous |         |           |          |             |     |         |         |     |
| ---------- | ------- | ---- | ------------------- | --- | --- | ---------- | ------- | --------- | -------- | ----------- | --- | ------- | ------- | --- |
|            |         |      |                     |     |     |            | S.; and | Hassabis, | D. 2015. | Human-level |     | control | through |     |
controlviaparameterfreezing.InAdvancesinNeuralInfor-
|     |     |     |     |     |     |     | deepreinforcementlearning. |     |     | Nat.,518(7540):529–533. |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | ----------------------- | --- | --- | --- | --- |
mationProcessingSystems34,23983–23992.
|     |     |     |     |     |     |     | Parisotto,E.;Ba,L.J.;andSalakhutdinov,R.2016. |     |     |     |     |     |     | Actor- |
| --- | --- | --- | --- | --- | --- | --- | --------------------------------------------- | --- | --- | --- | --- | --- | --- | ------ |
Buterez,D.;Janet,J.P.;Kiddle,S.J.;Oglic,D.;andLio`,P.
Mimic:DeepMultitaskandTransferReinforcementLearn-
2022. GraphNeuralNetworkswithAdaptiveReadouts. In ing. InProceedingsofthe4thInternationalConferenceon
AdvancesinNeuralInformationProcessingSystems. LearningRepresentations.
Chen, Y.; Chen, Y.; Yang, Y.; Li, Y.; Yin, J.; and Fan, C. Rajendran, J.; Lakshminarayanan, A. S.; Khapra, M. M.;
2019.LearningAction-TransferablePolicywithActionEm- Prasanna, P.; and Ravindran, B. 2017. Attend, Adapt and
bedding. CoRR,abs/1909.02291. Transfer:AttentiveDeepArchitectureforAdaptiveTransfer
Ferna´ndez, F.; and Veloso, M. M. 2006. Probabilistic pol- frommultiplesourcesinthesamedomain. InProceedings
icyreuseinareinforcementlearningagent. InProceedings of the 5th International Conference on Learning Represen-
tations.
| of the 5th | International |     | Joint Conference |     | on  | Autonomous |     |     |     |     |     |     |     |     |
| ---------- | ------------- | --- | ---------------- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
AgentsandMultiagentSystems,720–727.ACM. Rusu,A.A.;Colmenarejo,S.G.;Gu¨lc¸ehre,C¸.;Desjardins,
G.;Kirkpatrick,J.;Pascanu,R.;Mnih,V.;Kavukcuoglu,K.;
| Gilmer,    | J.; Schoenholz,                          |        | S. S.; Riley, | P.      | F.; Vinyals, | O.; and     |              |               |       |                      |     |          |             |     |
| ---------- | ---------------------------------------- | ------ | ------------- | ------- | ------------ | ----------- | ------------ | ------------- | ----- | -------------------- | --- | -------- | ----------- | --- |
|            |                                          |        |               |         |              |             | and Hadsell, | R.            | 2016. | Policy Distillation. |     | In       | Proceedings |     |
| Dahl, G.   | E. 2017.                                 | Neural | Message       | Passing |              | for Quantum |              |               |       |                      |     |          |             |     |
|            |                                          |        |               |         |              |             | of the 4th   | International |       | Conference           | on  | Learning | Represen-   |     |
| Chemistry. | InPrecup,D.;andTeh,Y.W.,eds.,Proceedings |        |               |         |              |             |              |               |       |                      |     |          |             |     |
tations.
ofthe34thInternationalConferenceonMachineLearning,
volume 70 of Proceedings of Machine Learning Research, Schmitt,S.;Hudson,J.J.;Z´ıdek,A.;Osindero,S.;Doersch,
|     |     |     |     |     |     |     | C.; Czarnecki, |     | W. M.; | Leibo, J. | Z.; Ku¨ttler, | H.; | Zisserman, |     |
| --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ------ | --------- | ------------- | --- | ---------- | --- |
1263–1272.PMLR.
|                            |            |           |                          |        |             |            | A.;Simonyan,K.;andEslami,S.M.A.2018. |             |     |               |                      |          | Kickstarting |     |
| -------------------------- | ---------- | --------- | ------------------------ | ------ | ----------- | ---------- | ------------------------------------ | ----------- | --- | ------------- | -------------------- | -------- | ------------ | --- |
| Gupta,                     | A.; Devin, | C.;       | Liu, Y.; Abbeel,         |        | P.; and     | Levine, S. |                                      |             |     |               |                      |          |              |     |
|                            |            |           |                          |        |             |            | DeepReinforcementLearning.           |             |     |               | CoRR,abs/1803.03835. |          |              |     |
| 2017. Learning             |            | Invariant | Feature                  | Spaces | to Transfer | Skills     |                                      |             |     |               |                      |          |              |     |
|                            |            |           |                          |        |             |            | Schulman,                            | J.; Wolski, |     | F.; Dhariwal, | P.;                  | Radford, | A.;          | and |
| withReinforcementLearning. |            |           | InProceedingsofthe5thIn- |        |             |            |                                      |             |     |               |                      |          |              |     |
Klimov,O.2017.ProximalPolicyOptimizationAlgorithms.
ternationalConferenceonLearningRepresentations.
CoRR,abs/1707.06347.
| Hong, S.;   | Yoon,  | D.; and | Kim, K.       | 2022. | Structure-Aware |       |             |        |     |           |     |           |     |        |
| ----------- | ------ | ------- | ------------- | ----- | --------------- | ----- | ----------- | ------ | --- | --------- | --- | --------- | --- | ------ |
|             |        |         |               |       |                 |       | Silver, D.; | Huang, | A.; | Maddison, | C.  | J.; Guez, | A.; | Sifre, |
| Transformer | Policy | for     | Inhomogeneous |       | Multi-Task      | Rein- |             |        |     |           |     |           |     |        |
L.;vandenDriessche,G.;Schrittwieser,J.;Antonoglou,I.;
InTheTenthInternationalConference
forcementLearning.
Panneershelvam,V.;Lanctot,M.;Dieleman,S.;Grewe,D.;
| on Learning | Representations, |     | ICLR | 2022, | Virtual | Event, |           |               |     |                |     |                |     |        |
| ----------- | ---------------- | --- | ---- | ----- | ------- | ------ | --------- | ------------- | --- | -------------- | --- | -------------- | --- | ------ |
|             |                  |     |      |       |         |        | Nham, J.; | Kalchbrenner, |     | N.; Sutskever, |     | I.; Lillicrap, |     | T. P.; |
April25-29,2022.
Leach,M.;Kavukcuoglu,K.;Graepel,T.;andHassabis,D.
Hu,Y.;Gao,Y.;andAn,B.2015a. AcceleratingMultiagent 2016. MasteringthegameofGowithdeepneuralnetworks
Reinforcement Learning by Equilibrium Transfer. IEEE andtreesearch. Nat.,529(7587):484–489.
Trans.Cybern.,45(7):1289–1302.
|     |     |     |     |     |     |     | Tao, Y.; | Genc, | S.; Chung, | J.; | Sun, | T.; and | Mallya, | S.  |
| --- | --- | --- | --- | --- | --- | --- | -------- | ----- | ---------- | --- | ---- | ------- | ------- | --- |
Hu,Y.;Gao,Y.;andAn,B.2015b. LearninginMulti-agent 2021. REPAINT: Knowledge Transfer in Deep Reinforce-
Systems with Sparse Interactions by Knowledge Transfer ment Learning. In Proceedings of the 38th International
ConferenceonMachineLearning,volume139ofProceed-
| and Game | Abstraction. |     | In Weiss, | G.; | Yolum, | P.; Bordini, |     |     |     |     |     |     |     |     |
| -------- | ------------ | --- | --------- | --- | ------ | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
R. H.; and Elkind, E., eds., Proceedings of the 2015 Inter- ingsofMachineLearningResearch,10141–10152.PMLR.
nationalConferenceonAutonomousAgentsandMultiagent Taylor,M.E.;andStone,P.2009. TransferLearningforRe-
Systems,753–761.ACM. inforcementLearningDomains:ASurvey. J.Mach.Learn.
Li,S.;andZhang,C.2018. AnOptimalOnlineMethodof Res.,10:1633–1685.
Selecting Source Policies for Reinforcement Learning. In Todorov, E.; Erez, T.; and Tassa, Y. 2012. MuJoCo: A
Proceedings of the Thirty-Second AAAI Conference on Ar- physicsengineformodel-basedcontrol. In2012IEEE/RSJ
tificialIntelligence,(AAAI-18),the30thinnovativeApplica- InternationalConferenceonIntelligentRobotsandSystems,
tions of Artificial Intelligence (IAAI-18), and the 8th AAAI 5026–5033.IEEE.
16359

TheThirty-EighthAAAIConferenceonArtificialIntelligence(AAAI-24)
Velicˇkovic´,P.;Cucurull,G.;Casanova,A.;Romero,A.;Lio`,
P.; and Bengio, Y. 2018. Graph Attention Networks. In
InternationalConferenceonLearningRepresentations.
Wan, M.; Gangwani, T.; and Peng, J. 2020. Mutual Infor-
mation Based Knowledge Transfer Under State-Action Di-
mensionMismatch. InProceedingsoftheThirty-SixthCon-
ferenceonUncertaintyinArtificialIntelligence,volume124
ofProceedingsofMachineLearningResearch,1218–1227.
AUAIPress.
Wang, T.; Liao, R.; Ba, J.; and Fidler, S. 2018. NerveNet:
LearningStructuredPolicywithGraphNeuralNetworks. In
Proceedings of the 6th International Conference on Learn-
ingRepresentations.
Yang, T.; Hao, J.; Meng, Z.; Zhang, Z.; Hu, Y.; Chen, Y.;
Fan, C.; Wang, W.; Liu, W.; Wang, Z.; and Peng, J. 2020a.
EfficientDeepReinforcementLearningviaAdaptivePolicy
Transfer. InProceedingsoftheTwenty-NinthInternational
JointConferenceonArtificialIntelligence,3094–3100.
Yang, T.; Hao, J.; Meng, Z.; Zhang, Z.; Hu, Y.; Chen, Y.;
Fan,C.;Wang,W.;Wang,Z.;andPeng,J.2020b. Efficient
Deep Reinforcement Learning through Policy Transfer. In
Proceedings of the 19th International Conference on Au-
tonomous Agents and Multiagent Systems, 2053–2055. In-
ternationalFoundationforAutonomousAgentsandMultia-
gentSystems.
Yang, T.; Wang, W.; Tang, H.; Hao, J.; Meng, Z.; Mao,
H.; Li, D.; Liu, W.; Chen, Y.; Hu, Y.; et al. 2021. An Ef-
ficient Transfer Learning Framework for Multiagent Rein-
forcement Learning. In Advances in Neural Information
ProcessingSystems,volume34.
You, H.; Yang, T.; Zheng, Y.; Hao, J.; and Taylor, M. E.
2022. Cross-domain Adaptive Transfer Reinforcement
Learning Based on State-Action Correspondence. In Pro-
ceedingsoftheThirty-eighthConferenceonUncertaintyin
ArtificialIntelligence.
Zhang, Q.; Xiao, T.; Efros, A. A.; Pinto, L.; and Wang, X.
2021. LearningCross-DomainCorrespondenceforControl
with Dynamics Cycle-Consistency. In Proceedings of the
9thInternationalConferenceonLearningRepresentations.
Zhu, Z.; Lin, K.; and Zhou, J. 2020. Transfer Learn-
ing in Deep Reinforcement Learning: A Survey. CoRR,
abs/2009.07888.
16360
