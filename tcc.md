IEEETRANSACTIONSONCLOUDCOMPUTING,VOL.XX,NO.X,2026 1
Cross-Domain DRL Agents for Efficient Job
Placement in the Cloud-Edge Continuum
Theodoros Aslanidis , Sokol Kosta , Spyros Lalis , Dimitris Chatzopoulos
Abstract—Thegrowingcomputationaldemandsofmodernap- often faces constraints in computational power, which caused
plicationscallforresourcemanagementstrategiesthateffectively the emergence of the cloud-edge continuum. This is a hy-
utilize the strengths of both cloud and edge computing. Deep
brid model that combines the strengths of both paradigms.
Reinforcement Learning (DRL) has shown great promise in
Thecloud-edgecontinuumcombinesrobust,scalableresource
addressing these challenges, offering advanced decision-making
capabilities that optimize resource allocation and system perfor- management with local, real-time processing, making it par-
mance.However,deployingDRLagentsincloud-edgecontinuum ticularly suited for applications in smart cities, healthcare,
infrastructuresremainsasignificantchallengeduetotheirdepen- and other domains where both centralized control and fast
denceoninfrastructure-specificstate-actionrepresentations.This
responsiveness are essential [4]. Job placement in cloud-edge
paper presents a novel architectural framework for DRL agents
continuum environments is a critical and recurring challenge,
that incorporates feature extraction and adaptation mechanisms
to enable their seamless operation across diverse environments. as continuous fluctuations in workload, resource availability,
By transforming state features into an infrastructure-agnostic andnetworkconditionsdemandrepeated,optimizeddecisions
representation, our approach reduces the need for extensive to maintain high service quality and efficient resource uti-
retraining when system configurations change. Experimental
lization [1], [3]. Moreover, while simple heuristics or rule-
results show that our method outperforms both a heuristic
based approaches may offer an initial solution, they are
method and a DRL baseline algorithm while achieving faster
convergence when infrastructure and workloads change. This generally inadequate for these complex systems because they
work is an important step forward in developing transferable cannot dynamically capture and adapt to the rapidly changing
andadaptableDRLsolutionsforreal-worldcloud-edgeresource operational conditions inherent in heterogeneous cloud-edge
management challenges.
infrastructures [4], [5].
Index Terms—Reinforcement Learning, Resource Allocation, The inherent complexity and heterogeneity of the cloud-
Cloud Computing, Cloud-Edge Continuum
edge continuum have motivated the integration of machine
learning techniques to optimize decision-making. Among
I. INTRODUCTION these, Reinforcement Learning (RL), and more specifically,
CLOUD computing has revolutionized the way comput- Deep Reinforcement Learning (DRL) has proven particularly
ing resources are provisioned, offering centralized, on- effective for managing dynamic, non-periodic user patterns
demandaccesstostorage,processingpower,andapplications. and making long-term, strategic decisions in such environ-
This paradigm enables flexible, scalable, and collaborative ments [6]–[10]. Various DRL paradigms, including hierarchi-
service deployment, making it a cornerstone of modern com- cal [6], multi-agent [11], and multi-objective RL [12], [13],
puting infrastructures [1]. However, the rapid proliferation have been explored to improve scalability and adaptability.
of Internet of Things devices and the resulting explosion Moreover, recent advances in meta-learning [14] and contin-
of data streams have exposed the limitations of centralized ual reinforcement learning [15] have demonstrated promising
cloud infrastructures, which increasingly struggle to handle approaches for reusing and retaining knowledge, enabling the
the computational load [2]. Edge computing addresses these development of more generalizable DRL agents in real-world
challengesbydistributingprocessingandstorageclosertodata scenarios [16]. These frameworks expose DRL models to
sources, reducing latency and bandwidth demands for real- a wide range of variances during training, promoting better
time applications [3]. Despite its advantages, edge computing generalizationtonewtaskswhileapplyingpreviouslyacquired
knowledgewithoutsufferingfromcatastrophicforgetting[17],
ManuscriptreceivedApril20,2026.Thisworkhasbeensupportedby(i) [18].
theHorizonEuroperesearchandinnovationprogramoftheEuropeanUnion,
MLSysOps (grant agreement number 101092912), and (ii) partly supported State of the art. These approaches mainly address variability
bytheCLEVERproject(grantagreementnumber101097560).CLEVERis
in workload distributions, ensuring robust performance across
supported by the EU Chips Joint Undertaking (Chips JU) and its members
(includingtop-upfundingbytheInnovationFundDenmark(IFD)). different data inputs. While adapting to input distributions
Theodoros Aslanidis and Dimitris Chatzopoulos are with the School is crucial for mitigating model degradation due to domain
of Computer Science, University College Dublin, Ireland (e-mail:
shift and concept drift [19], [20], DRL solutions in real-world
theodoros.aslanidis@ucdconnect.ie;dimitris.chatzopoulos@ucd.ie.
Sokol Kosta is with the Department of Electronic Systems, Aalborg scenarios face an even more pressing challenge: the tight
University,Denmark(e-mail:sok@es.aau.dk). coupling between the state-action space of an agent and the
SpyrosLalisiswiththeDepartmentofElectricalandComputerEngineer-
specificenvironmentinwhichitistrained[21]–[24].Whenthe
ing,UniversityofThessaly,Greece(e-mail:lalis@uth.gr).
DigitalObjectIdentifier:XX.XXXX/TCC.2026.XXXXXXX underlying infrastructure changes due to hardware upgrades,
0000–0000/00$00.00©2026IEEE

| 2   |     |     |     |     |     |     |     | IEEETRANSACTIONSONCLOUDCOMPUTING,VOL.XX,NO.X,2026 |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
network reconfigurations, or resource allocation adjustments, adaptation networks. While these methods improve conver-
the state-action space may also change, rendering the agent gence and adaptability, they assume that state-action spaces
unable to interact with the environment. remain consistent across tasks, limiting their applicability
Thisissue,knownasstate-actiondimensionmismatch[24], when infrastructure changes.
isafundamentalproblemincross-domaintransferanddomain This challenge is particularly acute in modern cloud-edge
| adaptation. | It requires | redesigning |     | and retraining |     | agents | from |           |                  |     |       |     |        |         |     |
| ----------- | ----------- | ----------- | --- | -------------- | --- | ------ | ---- | --------- | ---------------- | --- | ----- | --- | ------ | ------- | --- |
|             |             |             |     |                |     |        |      | computing | infrastructures, |     | which | are | highly | dynamic | and |
scratch whenever the infrastructure changes, a prohibitively heterogeneous. Variations in datacenter configurations driven
expensive process in terms of time and resources. by differences in hardware, network topologies, and resource
| Recent | works | have | focused | on transferring |     | knowledge |     |            |          |     |         |     |             |     |           |
| ------ | ----- | ---- | ------- | --------------- | --- | --------- | --- | ---------- | -------- | --- | ------- | --- | ----------- | --- | --------- |
|        |       |      |         |                 |     |           |     | allocation | policies | are | common, | as  | are changes |     | caused by |
across tasks with mismatched state-action spaces. For ex- equipment upgrades or failures. These variations pose sig-
| ample, | [21], | [25] propose |     | methods | that | learn | compact |          |            |     |     |         |          |         |        |
| ------ | ----- | ------------ | --- | ------- | ---- | ----- | ------- | -------- | ---------- | --- | --- | ------- | -------- | ------- | ------ |
|        |       |              |     |         |      |       |         | nificant | challenges | for | DRL | agents, | as those | trained | on one |
latent representations— via autoencoders or mutual infor- infrastructureoftenstruggletogeneralizetoothers,ormaybe-
mation objectives—to disentangle task-specific details from come entirely incompatible. Without mechanisms for efficient
| generalizable | features. |     | Similarly, | [24], | [26] | construct | em- |             |     |            |     |         |            |              |     |
| ------------- | --------- | --- | ---------- | ----- | ---- | --------- | --- | ----------- | --- | ---------- | --- | ------- | ---------- | ------------ | --- |
|               |           |     |            |       |      |           |     | adaptation, | DRL | approaches |     | that do | not handle | cross-domain |     |
bedding spaces for policy transfer across domains. While transfers and state-action space mismatches become obsolete,
thesemethodshaveshownpromiseinstructuredenvironments requiring new state-action space designs and retraining from
such as robotics (e.g., MuJoCo [27] locomotion tasks), they scratch with every system change.
| rely on well-defined |                 | state-action  |               | representations.             |            | It             | is im- |                |              |                  |                 |           |                          |                 |          |
| -------------------- | --------------- | ------------- | ------------- | ---------------------------- | ---------- | -------------- | ------ | -------------- | ------------ | ---------------- | --------------- | --------- | ------------------------ | --------------- | -------- |
|                      |                 |               |               |                              |            |                |        | Contributions. |              | To address       | the             | challenge |                          | of state-action | di-      |
| portant to           | note            | that although |               | dynamic conditions—including |            |                |        |                |              |                  |                 |           |                          |                 |          |
|                      |                 |               |               |                              |            |                |        | mension        | mismatch,    | we               | propose         | a novel   | framework                |                 | that de- |
| fluctuating          | constraints     | and           | communication |                              | delays—are |                | also   |                |              |                  |                 |           |                          |                 |          |
|                      |                 |               |               |                              |            |                |        | couples        | DRL          | agents from      | domain-specific |           |                          | features,       | enabling |
| present in           | robotics,       | the           | nature        | and scale                    | of these   | challenges     |        |                |              |                  |                 |           |                          |                 |          |
|                      |                 |               |               |                              |            |                |        | them to        | operate      | effectively      | across          | diverse   | cloud-edge               |                 | environ- |
| in cloud-edge        | infrastructures |               |               | are fundamentally            |            | different.     | In     |                |              |                  |                 |           |                          |                 |          |
|                      |                 |               |               |                              |            |                |        | ments.         | Our approach | employs          |                 | state     | abstraction              | techniques      | to       |
| robotics,            | environmental   |               | variations    | typically                    | stem       | from           | con-   |                |              |                  |                 |           |                          |                 |          |
|                      |                 |               |               |                              |            |                |        | create a       | unified,     | fixed-dimension, |                 | and       | infrastructure-invariant |                 |          |
| trolled physical     |                 | interactions  | and           | predictable                  |            | task dynamics, |        |                |              |                  |                 |           |                          |                 |          |
representationofthestatespace.Thediscretestatefeaturesare
whereascloud-edgesystemsmustcontendwithhighlyvolatile
mappedintofixed-sizeembeddingsthatcaptureunderlyingre-
| resource | availability, | heterogeneous |     | hardware |     | configurations, |     |     |     |     |     |     |     |     |     |
| -------- | ------------- | ------------- | --- | -------- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
and complex, multi-hop network topologies. Consequently, lationshipsinadense,continuousspace.Then,thecontinuous
anddiscretefeaturesarepassedthroughtwodistincttwo-layer
| while cross-domain |           | RL          | approaches    | from          | robotics |          | provide |                |          |          |              |         |                |                |            |
| ------------------ | --------- | ----------- | ------------- | ------------- | -------- | -------- | ------- | -------------- | -------- | -------- | ------------ | ------- | -------------- | -------------- | ---------- |
|                    |           |             |               |               |          |          |         | MLP networks   |          | and then | concatenated |         | and            | passed         | through an |
| valuable           | insights, | they do     | not           | fully address | the      | unique   | state-  |                |          |          |              |         |                |                |            |
|                    |           |             |               |               |          |          |         | adaptive       | residual | layer,   | which        | further | enhances       | generalization |            |
| action mismatches  |           | and         | the extensive | variability   |          | inherent |         | in             |          |          |              |         |                |                |            |
|                    |           |             |               |               |          |          |         | and adaptation |          | during   | transfers.   | This    | transformation |                | abstracts  |
| cloud-edge         | resource  | management. |               |               |          |          |         |                |          |          |              |         |                |                |            |
Other works in cross-domain RL have explored policy infrastructure-specific details, enabling effective knowledge
|                             |         |            |              |                             |           |     |           | transfer | across      | different     | cloud-edge |                  | environments |      | without the |
| --------------------------- | ------- | ---------- | ------------ | --------------------------- | --------- | --- | --------- | -------- | ----------- | ------------- | ---------- | ---------------- | ------------ | ---- | ----------- |
| representation              | and     | transfer   | in           | more heterogeneous          |           |     | settings. |          |             |               |            |                  |              |      |             |
|                             |         |            |              |                             |           |     |           | need for | agent       | redesign.     |            |                  |              |      |             |
| For example,                |         | [28], [29] | propose      | techniques                  |           | for | adapting  |          |             |               |            |                  |              |      |             |
|                             |         |            |              |                             |           |     |           | Our      | work        | is orthogonal |            | to meta-learning |              | and  | continual   |
| policiesacrossdomains,while |         |            |              | [24],[30]extendtheseideasby |           |     |           |          |             |               |            |                  |              |      |             |
|                             |         |            |              |                             |           |     |           | learning | approaches. |               | Once       | the agent        | is           | made | compatible  |
| integrating                 | learned | state      | abstractions | and                         | knowledge |     | transfer  |          |             |               |            |                  |              |      |             |
mechanisms.However,thesemethodsoftenassumethatsource withmultipleinfrastructures,ourframeworkcanbecombined
|            |       |            |            |           |     |                |     | with meta-learning |     | or continual |     | learning | techniques |     | to further |
| ---------- | ----- | ---------- | ---------- | --------- | --- | -------------- | --- | ------------------ | --- | ------------ | --- | -------- | ---------- | --- | ---------- |
| and target | tasks | share some | underlying | structure |     | or similarity, |     |                    |     |              |     |          |            |     |            |
which may not hold in practical scenarios like cloud-edge enhance adaptability to changes in input workloads. This
|                      |          |               |          |                   |              |       |        | integration | would      | create | DRL            | agents | capable        | of  | generalizing |
| -------------------- | -------- | ------------- | -------- | ----------------- | ------------ | ----- | ------ | ----------- | ---------- | ------ | -------------- | ------ | -------------- | --- | ------------ |
| resource management. |          | Additionally, |          | while             | disentangled |       | repre- |             |            |        |                |        |                |     |              |
|                      |          |               |          |                   |              |       |        | not only    | to changes | in     | the underlying |        | infrastructure |     | but also     |
| sentation            | learning | [31],         | [32] and | autoencoder-based |              | state | rep-   |             |            |        |                |        |                |     |              |
resentation learning [33]–[35], have been proposed to capture to variations in input distributions.
|                  |     |           |      |           |       |     |          | This | work | bridges the | gap | between | DRL | models | and their |
| ---------------- | --- | --------- | ---- | --------- | ----- | --- | -------- | ---- | ---- | ----------- | --- | ------- | --- | ------ | --------- |
| domain-invariant |     | features, | they | primarily | focus | on  | robotics |      |      |             |     |         |     |        |           |
or simulation settings. These approaches do not address practical deployment in dynamic, heterogeneous cloud-edge
|            |            |     |            |                  |     |     |         | systems. | Our | contributions | can | be summarized |     | as  | follows: |
| ---------- | ---------- | --- | ---------- | ---------------- | --- | --- | ------- | -------- | --- | ------------- | --- | ------------- | --- | --- | -------- |
| the unique | challenges | of  | cloud-edge | infrastructures, |     |     | such as |          |     |               |     |               |     |     |          |
fluctuating resource constraints, dynamic network topologies, • Weformulatethejobplacementprobleminmulti-datacenter
and hardware heterogeneity, which significantly impact RL infrastructures as an RL task and propose a state-action space
adaptation. and reward design for multi-datacenter cloud-edge resource
Some works have explored transfer learning in the con- management, enabling seamless cross-domain transfer and
text of cloud-edge environments but do not address state- adaptation across environments with varying scales.
action mismatches caused by infrastructure changes. For ex- • We integrate a custom feature extractor into the internal ar-
ample, [36] proposes a framework for cloud-edge collab- chitecture of the DRL agent, using state abstraction and adap-
orative DRL, focusing on knowledge distillation between tation techniques to decouple the agent from infrastructure-
heterogeneous agents. Similarly, [37] introduces a transfer specific details. This approach enables rapid adaptation to
RL framework for adaptive task offloading, using domain infrastructure changes.
adaptation to align heterogeneous characteristics of mobile • We develop a framework for job-to-datacenter placement
devices. Finally, [38] proposes a hybrid cloud-edge control and evaluate it through extensive simulations. We compare
strategyusingtransferDRL,relyingonfine-tuninganddomain it against a heuristic and a baseline DRL approach, demon-

ASLANIDISetal.:CROSS-DOMAINDRLAGENTSFOREFFICIENTJOBPLACEMENTINTHECLOUD-EDGECONTINUUM 3
strating significant improvements in terms of total reward and fluctuating resource constraints, dynamic network topologies,
convergence speed. and hardware heterogeneity, which significantly impact RL
| To the | best of our | knowledge, |     | no prior | work | addresses the | adaptation. |     |     |     |     |     |     |     |
| ------ | ----------- | ---------- | --- | -------- | ---- | ------------- | ----------- | --- | --- | --- | --- | --- | --- | --- |
challenge of state-action dimension mismatch in the cloud- Transfer RL for Cloud-Edge Resource Management. In
edgecontinuumenvironmentsforresourcemanagementtasks. the context of cloud-edge environments, some works have
Existing methods are either highly theoretical and impractical explored transfer learning but do not address state-action mis-
for real-world deployment or tailored to specific domains like matches caused by infrastructure changes. For example, [36]
robotics or games. This challenge is particularly critical in proposeaframeworkforcloud-edgecollaborativeDRL,focus-
cloud-edge environments, where infrastructure changes are ing on knowledge distillation between heterogeneous agents.
common. Our work fills this gap by proposing a transfer Similarly,[37]introduceatransferRLframeworkforadaptive
learning framework that uses state abstraction to create an task offloading, using domain adaptation to align heteroge-
infrastructure-invariant representation, enabling seamless pol- neouscharacteristicsofmobiledevices.Finally,[38]proposea
icy transfer across diverse environments. By decoupling the hybridcloud-edgecontrolstrategyusingtransferDRL,relying
agentfromdomain-specificfeatures,ourframeworkminimizes on fine-tuning and domain adaptation networks. While these
theneedforextensiveretraining,ensuringrobustperformance methods improve convergence and adaptability, they assume
across varying system configurations. thatstate-actionspacesremainconsistentacrosstasks,limiting
The rest of the paper is organized as follows. First, in their applicability when infrastructure changes.
Section II, we provide an overview of related work. Then, To the best of our knowledge, no prior work addresses the
|     |     |     |     |     |     |     | challenge | of state-action |     | dimension |     | mismatch | in  | the cloud- |
| --- | --- | --- | --- | --- | --- | --- | --------- | --------------- | --- | --------- | --- | -------- | --- | ---------- |
inSectionIII,wecovertheprelimirariesneededforthebetter
understanding of this work. In Section IV, we present the edgecontinuumenvironmentsforresourcemanagementtasks.
system model and the formulation of the problem. These are Existing methods are either highly theoretical and impractical
followed by the solution methods in Sections V and VI-A. for real-world deployment or tailored to specific domains like
Then, in Section V we present extensions of our system. Sec- robotics or games. This challenge is particularly critical in
tion VI presents our experimental setup and the performance cloud-edge environments, where infrastructure changes are
evaluation of our approach. Section VII discusses and finally, common. Our work fills this gap by proposing a transfer
Section VIII concludes the article. learning framework that uses state abstraction to create an
|     |     |                 |     |     |     |     | infrastructure-invariant |        |         | representation, |     | enabling | seamless   | pol- |
| --- | --- | --------------- | --- | --- | --- | --- | ------------------------ | ------ | ------- | --------------- | --- | -------- | ---------- | ---- |
|     |     |                 |     |     |     |     | icy transfer             | across | diverse | environments.   |     | By       | decoupling | the  |
|     |     | II. RELATEDWORK |     |     |     |     |                          |        |         |                 |     |          |            |      |
agentfromdomain-specificfeatures,ourframeworkminimizes
Cross-DomainAdaptationandTransfer.Anumberofrecent theneedforextensiveretraining,ensuringrobustperformance
workshavefocusedonthechallengeoftransferringknowledge across varying system configurations.
| across tasks      | with mismatched |             | state-action  |                       | spaces,   | a common     |           |             |      |            |          |              |                 |           |
| ----------------- | --------------- | ----------- | ------------- | --------------------- | --------- | ------------ | --------- | ----------- | ---- | ---------- | -------- | ------------ | --------------- | --------- |
| issue in          | cross-domain    | RL.         | For           | example,              | [21],     | [25] propose |           |             |      |            |          |              |                 |           |
|                   |                 |             |               |                       |           |              |           |             | III. | BACKGROUND |          |              |                 |           |
| methods           | that learn      | compact     | latent        | representations—often |           | via          |           |             |      |            |          |              |                 |           |
|                   |                 |             |               |                       |           |              | RL. The   | RL paradigm |      | includes   | an agent | and          | an environment. |           |
| autoencoders      | or mutual       | information |               | objectives—to         |           | disentan-    |           |             |      |            |          |              |                 |           |
|                   |                 |             |               |                       |           |              | The agent | observes    | the  | state s    | of the   | environment, |                 | interacts |
| gle task-specific | details         | from        | generalizable |                       | features. | Simi-        |           |             |      |            |          |              |                 |           |
larly, [24], [26] address state-action mismatches by construct- with it by taking actions a based on a learned behavior,
|               |        |      |            |        |          |            | formally    | called | a policy, | and         | receives | feedback | in               | the form |
| ------------- | ------ | ---- | ---------- | ------ | -------- | ---------- | ----------- | ------ | --------- | ----------- | -------- | -------- | ---------------- | -------- |
| ing embedding | spaces | that | facilitate | policy | transfer | across do- |             |        |           |             |          |          |                  |          |
|               |        |      |            |        |          |            | of a reward | signal | r.        | Each action |          | affects  | the environment, |          |
mains.Whilethesemethodshaveshownpromiseinstructured
environments such as robotics (e.g., MuJoCo [27] locomotion causing a transition to a new state. These interactions occur
|              |         |              |     |              |                  |     | over discrete | time | steps | t, which | together |     | form | an episode. |
| ------------ | ------- | ------------ | --- | ------------ | ---------------- | --- | ------------- | ---- | ----- | -------- | -------- | --- | ---- | ----------- |
| tasks), they | rely on | well-defined |     | state-action | representations. |     |               |      |       |          |          |     |      |             |
However, in cloud-edge resource management tasks, such The goal of the agent is to maximize its cumulative episodic
|              |                 |              |               |                |          |              | reward by | discovering  | an         | optimal  | policy.   | RL         | has been     | widely    |
| ------------ | --------------- | ------------ | ------------- | -------------- | -------- | ------------ | --------- | ------------ | ---------- | -------- | --------- | ---------- | ------------ | --------- |
| standardized | representations |              | are           | not available, |          | making these |           |              |            |          |           |            |              |           |
|              |                 |              |               |                |          |              | applied   | in robotics, | game       | playing, | and       | autonomous |              | decision- |
| methods      | less suitable   | for such     | environments. |                |          |              |           |              |            |          |           |            |              |           |
|              |                 |              |               |                |          |              | making,   | where        | sequential | decision | processes |            | are crucial. |           |
| Other        | works in        | cross-domain |               | RL have        | explored | policy       |           |              |            |          |           |            |              |           |
representationandtransferinmoreheterogeneoussettings.For DRL.DRLextendsstandardRLbyincorporatingdeepneural
example, [28], [29] propose techniques for adapting policies networksasfunctionapproximatorstolearncomplexpolicies,
across domains, while [24], [30] and extend these ideas by enabling agents to handle large, high-dimensional state-action
integrating learned state abstractions and knowledge transfer spaces found in modern systems. However, this comes at
mechanisms.However,thesemethodsoftenassumethatsource a cost, as training deep networks requires substantial com-
and target tasks share some underlying structure or similarity, putational resources, extensive environment interaction, and
which may not hold in practical scenarios like cloud-edge often suffers from instability during learning. The challenge
resource management. Additionally, while disentangled repre- of sample inefficiency arises, where an agent may require a
sentation learning [31], [32] and autoencoder-based state rep- largenumberofinteractionstoconvergetoaneffectivepolicy.
resentation learning [33]–[35], have been proposed to capture Additionally, a fundamental aspect of RL is the exploration-
domain-invariant features, they primarily focus on robotics exploitation tradeoff, where an agent must balance between
or simulation settings. These approaches do not address exploring new actions to discover better strategies and ex-
the unique challenges of cloud-edge infrastructures, such as ploitingknownactionstomaximizerewards.Poorexploration

| 4   |     |     |     |     |     |     |     | IEEETRANSACTIONSONCLOUDCOMPUTING,VOL.XX,NO.X,2026 |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
can lead to suboptimal policies, excessive training times, or and adapt to new environments efficiently, even when the
convergence to local optima. underlying infrastructure or task configuration changes. For
|              |             |        |       |            |     |         |          | example, | in cloud-edge  |     | resource     | management, | where          |     | infras- |
| ------------ | ----------- | ------ | ----- | ---------- | --- | ------- | -------- | -------- | -------------- | --- | ------------ | ----------- | -------------- | --- | ------- |
| Actor-Critic | Algorithms. |        | These | algorithms |     | consist | of two   |          |                |     |              |             |                |     |         |
|              |             |        |       |            |     |         |          | tructure | configurations |     | vary widely, | a unified   | representation |     |         |
| components:  | the         | actor, | which | determines | the | agent’s | actions, |          |                |     |              |             |                |     |         |
and the critic, which evaluates the quality of those actions of resource states (e.g., CPU usage, memory availability)
|              |           |     |         |          |       |     |          | enables | the agent | to generalize |     | across different |     | datacenters |     |
| ------------ | --------- | --- | ------- | -------- | ----- | --- | -------- | ------- | --------- | ------------- | --- | ---------------- | --- | ----------- | --- |
| by providing | feedback. |     | This is | achieved | using | two | separate |         |           |               |     |                  |     |             |     |
neural networks—one for each component. The state serves without requiring extensive retraining. This approach not only
|              |         |           |             |              |     |        |          | improves   | sample | efficiency | but               | also enhances |               | the agent’s |     |
| ------------ | ------- | --------- | ----------- | ------------ | --- | ------ | -------- | ---------- | ------ | ---------- | ----------------- | ------------- | ------------- | ----------- | --- |
| as input     | to both | the actor | and         | the critic.  | The | policy | network  |            |        |            |                   |               |               |             |     |
|              |         |           |             |              |     |        |          | ability to | handle | dynamic    | and heterogeneous |               | environments, |             |     |
| (actor) maps | states  | to a      | probability | distribution |     | over   | possible |            |        |            |                   |               |               |             |     |
actions.Itisresponsiblefordecision-making,selectingactions making it a cornerstone of effective cross-domain transfer
learning.
basedonlearnedpoliciesthatmaximizeexpectedrewards.The
policy is continuously refined using feedback from the critic, Embeddings in Policy Networks of DRL Agents. Em-
allowingtheagenttoexploreandexploiteffectively.Thevalue beddings (first introduced in [39], popularized in [40]) are
network(critic)estimatesascalarstatevalue,whichrepresents low-dimensionalrepresentationsofhigh-dimensionaldatathat
theexpectedcumulativerewardfromagivenstate.Itevaluates capture essential features while preserving meaningful rela-
the agent’s performance and computes the advantage, defined tionships. They replace raw one-hot or categorical encodings
as the difference between the predicted state value and the by mapping discrete variables into continuous vector spaces,
actual received reward. This advantage function helps guide allowing models to generalize more effectively. In DRL,
policy updates, improving decision-making over time. By embeddingsplayacrucialroleinpolicynetworksbyenabling
training both networks simultaneously, actor-critic algorithms agents to efficiently process large and complex state-action
enable more stable learning, balancing long-term planning spaces.Bylearningcompact,transferablerepresentations,em-
with immediate reward feedback, making them effective for beddings facilitate cross-domain policy adaptation, ensuring
complex reinforcement learning tasks. smoother transfer learning and better alignment of state-
|          |        |              |     |        |     |             |      | action spaces |            | in heterogeneous | environments. |     | This       | ability  | to  |
| -------- | ------ | ------------ | --- | ------ | --- | ----------- | ---- | ------------- | ---------- | ---------------- | ------------- | --- | ---------- | -------- | --- |
| Proximal | Policy | Optimization |     | (PPO). | PPO | is a widely | used |               |            |                  |               |     |            |          |     |
|          |        |              |     |        |     |             |      | encode        | meaningful | feature          | relationships | is  | especially | valuable |     |
actor-criticalgorithmthatoptimizesthepolicyusingaclipped
surrogate objective, which prevents excessively large updates in dynamic systems where traditional representations struggle
|     |     |     |     |     |     |     |     | to adapt | efficiently. |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ------------ | --- | --- | --- | --- | --- | --- |
thatcoulddestabilizetraining.Byconstrainingpolicyupdates,
PPO ensures a more stable learning process while effectively Residual Connections in Policy Networks of DRL Agents.
balancingexplorationandexploitation.Thesepropertiesmake
|     |     |     |     |     |     |     |     | Residual | connections | in  | neural networks, |     | introduced | in  | [41], |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ----------- | --- | ---------------- | --- | ---------- | --- | ----- |
PPO well-suited for complex DRL tasks that require reliable enable deeper architectures by allowing gradients to flow
performance and sample efficiency. more effectively through layers, mitigating vanishing gradient
Transfer RL & Cross-Domain Transfer. Transfer RL lever- issues. In DRL, residual connections in the policy network
|                 |     |       |           |      |          |     |         | facilitate | smoother | policy | updates | by preserving |     | useful | repre- |
| --------------- | --- | ----- | --------- | ---- | -------- | --- | ------- | ---------- | -------- | ------ | ------- | ------------- | --- | ------ | ------ |
| ages pretrained |     | agent | knowledge | from | a source |     | task to | a          |          |        |         |               |     |        |        |
sentationswhileenablingadaptationtonewtasks.Fortransfer
| target task | to reduce |     | the required | interactions, |     | significantly |     |     |     |     |     |     |     |     |     |
| ----------- | --------- | --- | ------------ | ------------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
accelerating convergence and improving sample efficiency. In learning, residual connections help retain previously learned
|              |            |     |            |      |          |       |           | knowledge | while | integrating | new | information, |     | preventing |     |
| ------------ | ---------- | --- | ---------- | ---- | -------- | ----- | --------- | --------- | ----- | ----------- | --- | ------------ | --- | ---------- | --- |
| cross-domain | transfers, |     | the source | task | that the | agent | initially |           |       |             |     |              |     |            |     |
trained on and the target task that the agent is later deployed catastrophic forgetting. When adapting to new policies across
|         |           |              |     |         |         |           |     | different     | environments, |                | residual | layers allow | for   | incremental |     |
| ------- | --------- | ------------ | --- | ------- | ------- | --------- | --- | ------------- | ------------- | -------------- | -------- | ------------ | ----- | ----------- | --- |
| on have | different | state-action |     | spaces. | The key | challenge |     | in            |               |                |          |              |       |             |     |
|         |           |              |     |         |         |           |     | modifications |               | to the policy, | ensuring | stability    | while | enabling    |     |
cross-domaintransferisenablingtheagenttoeffectivelyreuse
knowledge from tasks that do not share identical state-action flexibility in handling state-action mismatches during cross-
|     |     |     |     |     |     |     |     | domain | transfers. |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | ---------- | --- | --- | --- | --- | --- | --- |
representations,movingbeyondconventionaltransferlearning
| that assumes | a common |     | structure. | Various | approaches |     | in the |     |                                 |     |     |     |     |     |     |
| ------------ | -------- | --- | ---------- | ------- | ---------- | --- | ------ | --- | ------------------------------- | --- | --- | --- | --- | --- | --- |
|              |          |     |            |         |            |     |        |     | IV. PROBLEMFORMULATION&SOLUTION |     |     |     |     |     |     |
literatureaimtobridgethisgap,facilitatingknowledgetransfer
| across environments |     | with | different | state-action |     | spaces. |     |     |     |     |     |     |     |     |     |
| ------------------- | --- | ---- | --------- | ------------ | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
SystemModel.Weconsiderahierarchicalinfrastructurecom-
State Abstraction and Dimensionality Reduction. In re- posed of n datacenters, as illustrated in Figure 1. Datacenters
inforcement learning, the state space often contains high- are classified into three types: cloud, edge, and far-edge. For
dimensional, redundant, or irrelevant information that can instance, far-edge (on-premise) datacenters may be located at
hinder learning and generalization. State abstraction and di- two different universities in Dublin (e.g., UCD and DCU), an
mensionality reduction techniques address this challenge by edgedatacenterelsewhereinDublin,andaclouddatacenterin
transforming raw state features into a fixed, low-dimensional, Rotterdam. Each datacenter consists of multiple hosts. In our
and unified representation that captures the essential aspects scenario, all datacenters are interconnected except for the two
of the environment. This abstraction process is critical for far-edge datacenters, which remain isolated from one another.
enabling domain-invariant and environment-agnostic repre- Cloud datacenters typically offer a larger number of hosts
sentations, which allow agents to transfer knowledge across with more powerful machines, whereas edge and far-edge
environments with different state-action dimensionalities. By datacenters have fewer and less capable hosts but are acces-
decoupling the agent’s policy from environment-specific de- sible with a lower latency since they are closer to the users.
tails, state abstraction ensures that the agent can interact with Although each datacenter maintains its own job queue, we

ASLANIDISetal.:CROSS-DOMAINDRLAGENTSFOREFFICIENTJOBPLACEMENTINTHECLOUD-EDGECONTINUUM 5
Datacenter Job Latency
QoS
Type Tolerance
Cloud
Rotterdam Far-Edge Tolerant 1.0
Edge Tolerant 1.0
Cloud Tolerant 1.0
Far-Edge Moderate 1.0
Edge Edge Moderate 1.0
Dublin Cloud Moderate 0.0
Far-edge Critical 1.0
Edge Critical 0.5
Cloud Critical 0.0
Far-Edge TABLEI
UCD DCU JOBQOSACHIEVEDBASEDONDATACENTERTYPEANDJOBLATENCY
TOLERANCE.
Job Queue
and the current job queue. This visibility is based on a
lightweight state representation, where each host is described
Fig.1. Amulti-datacenterinfrastructureconsistingofonecloud,oneedge,
andtwofar-edgedatacenters(EnvB). by three numerical values and each job in the queue by four
numerical values (see Figure 2). The agent does not have
insight into future job arrivals.
preprocess these individual queues to create an aggregated
Theagentoperatesinaslottedmanner(withoneslotcorre-
view that forms part of the RL agent’s state space.
sponding to one second). It receives the current infrastructure
In our model, jobs arrive exclusively at far-edge locations
and job queue states at each slot and returns an action vector.
and may be placed in the same far-edge datacenter, in an
Whennojobsarepending,allactionsdefaulttono-ops.Since
edge datacenter connected to it, or, as a last resort, in a
the number of arriving jobs is variable, we define a maximum
cloud datacenter. Notably, jobs arriving at a particular far-
queue length to accommodate fluctuations in workload. We
edge(e.g.,UCD)cannotbedeployedtoanotherfar-edge(e.g.,
need to set a maximum job queue length to define the agent’s
DCU).EachjobischaracterizedbyitsCPUcorerequirements,
action space, which corresponds to placement decisions for
execution length, arrival location (corresponding to one of the
each job. This fixed limit is essential for the DRL agent’s
datacenters),asoftdeadline,andalatencytoleranceclassified
architecture, as it affects the output layer size of the neural
as tolerant, moderate, or critical.
network. The queue does not need to reach this maximum; if
The soft deadline defines a preferred timeframe for job
fewer jobs are present, slots are filled with zeros.
placement, whereas the latency tolerance determines how
State Space. The state space consists of two subspaces.
far from its arrival location the job can be placed without
The first represents the infrastructure load, where each host
significantdegradation(e.g.,closeratthefar-edge,attheedge,
in a datacenter is described by the tuple ⟨DC , DC ,
or further away in the cloud). Consequently, job placement id type
Host ⟩. Here, DC uniquely identifies a datacenter,
directly impacts QoS, as jobs processed farther from their freeCores id
DC indicates whether it is cloud, edge, or far-edge, and
arrivallocationexperiencehighercommunicationdelays(e.g., type
Host indicate the number of available cores at the this
longer round-trip times). To clarify how these parameters im- freeCores
slot. This subspace informs the agent about the currently
pact service quality, Table I summarizes the expected quality-
available computing resources. The second subspace captures
of-service (QoS) for each combination of datacenter type
the job queue, with each job represented by the tuple ⟨cores,
and job latency tolerance. The soft deadline represents the
location, tolerance, deadline⟩. The overall state at each
maximum allowable waiting time before placement. If the
timestep is formed by concatenating the infrastructure state
deadlineisexceeded,thejobisstillexecuted,butthisoutcome
with the job queue information, as depicted in Figure 2.
is considered suboptimal as it results in a degradation of the
job’s QoS. Action Space. The agent’s action is expressed as a vector
InourRLsetup,eachepisodeisdefinedbyafixedworkload whoselengthequalsthemaximumnumberofjobsthatcanbe
size. Specifically, we introduce a predefined number of jobs queued. Each element of this vector specifies the datacenter
(e.g., 50) into the system. The episode terminates once all (viaitsid)towhichthecorrespondingjobshouldbeassigned.
these jobs have been successfully executed, after which the To handle DRL agent transfers across infrastructures with
simulation resets, and the experiment is repeated. different numbers of datacenters, the action vector is defined
The primary objective of our DRL agent is to serve as a based on a user-specified maximum number of datacenters.
broker that maps incoming jobs to suitable datacenters for If an action is infeasible—such as assigning a job to a
execution. The job-to-host assignment within a datacenter is datacenter with insufficient free resources or a non-existent
determined by a heuristic that allocates the job to the host datacenter—the environment treats that action as a no-op.
with the maximum available resources. Furthermore, jobs are Similarly, if the agent issues an action for a job that is not
executedwithinvirtualmachines(VMs).Weassumethateach present (due to variable queue lengths), it is ignored.
hosthasanalways-onVMthatisavailableprovidedsufficient Reward Function. Our reward function integrates multiple
resources exist. The agent has full visibility of all datacenters components to balance system performance metrics. First, to

| 6   |     |     |     |     |     |     |     | IEEETRANSACTIONSONCLOUDCOMPUTING,VOL.XX,NO.X,2026 |         |       |        |             |                 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | ------- | ----- | ------ | ----------- | --------------- | --- | --- |
|     |     |     |     |     |     |     |     | transforms                                        | the raw | state | into a | compressed, | fixed-dimension |     |     |
Environment
|     | Job Queue |     |     |     | Infrastructure |     |     |                       |     |          |     |        |         |         |        |
| --- | --------- | --- | --- | --- | -------------- | --- | --- | --------------------- | --- | -------- | --- | ------ | ------- | ------- | ------ |
|     |           |     |     |     |                |     |     | latent representation |     | as shown | in  | Figure | 3. This | unified | latent |
Job1 Place  W aiting C l o u d representationisinfrastructure-agnostic,enablingtheagentnot
|     |     |     |     | Jo bs |     | Dat a | c e n ters |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ----- | --- | ----- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
Job2 only to learn effectively in a single environment but also to
Edge
Translate retain and transfer knowledge to similar environments.
|     |     |     | Action |     |     | Datacenters |     |           |             |     |         |          |     |          |        |
| --- | --- | --- | ------ | --- | --- | ----------- | --- | --------- | ----------- | --- | ------- | -------- | --- | -------- | ------ |
|     |     |     |        |     |     |             |     | The state | abstraction |     | process | operates | as  | follows. | First, |
...
Calculate Far-Edge all features in the raw observation are padded with zeros
|     |     | Jobn |     | Reward |     | Datacenters |     |             |             |              |             |           |             |               |        |
| --- | --- | ---- | --- | ------ | --- | ----------- | --- | ----------- | ----------- | ------------ | ----------- | --------- | ----------- | ------------- | ------ |
|     |     |      |     |        |     |             |     | up to their | maximum     | user-defined |             | values.   | Within      | the           | policy |
|     |     |      |     |        |     |             |     | network,    | continuous  | and          | categorical |           | features    | are processed |        |
|     |     |      |     |        |     |             |     | separately. | Categorical | features,    |             | typically | represented | via           | one-   |
Job Queue + Infrastructure hot encoding, are passed through an embedding layer to
|     |     | State |     |     | State |     |     |     |     |     |     |     |     |     |     |
| --- | --- | ----- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
State compactly capture their underlying relationships. Each fea-
|     |     |     | Action | Reward |     |     |     |               |            |     |                 |     |         |           |     |
| --- | --- | --- | ------ | ------ | --- | --- | --- | ------------- | ---------- | --- | --------------- | --- | ------- | --------- | --- |
|     |     |     |        |        |     |     |     | ture, whether | continuous |     | or categorical, |     | is then | processed |     |
DRL Agent
|     |     |     |     |     |     |     |     | by its own | dedicated | MLP | with | an  | identical | architecture. |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --------- | --- | ---- | --- | --------- | ------------- | --- |
Fig.2. TheDRLagenttakestheconcatenatedstatesasinput,selectsanaction, These MLPs use ReLU activation functions and dropout
and receives a reward based on the environment’s job placement response. for regularization. Their outputs are then concatenated and
Thislooprepeatsuntiltheepisodeends.
passedthroughanadaptationlayer—implementedasaresidual
|     |     |     |     |     |     |     |     | connection—before |     | action | selection. |     | This layer | allows | the |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | ------ | ---------- | --- | ---------- | ------ | --- |
encourage prompt job placement and prevent queue conges- agenttopreservepreviouslylearnedknowledgewhileadapting
tion, we define a placement reward, to new information, mitigating catastrophic forgetting. By
|     |     |     |     |     |     |     |     | converting | raw | observations | into | a unified, | domain-invariant |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | ------------ | ---- | ---------- | ---------------- | --- | --- |
jobsPlaced
|     |     |     | R = |     | .   |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
p jobsWaiting representation,theagentcantransferknowledgeacrossdiverse
|           |            |            |     |              |            |             |          | cloud-edge            | environments |                | without      | requiring | extensive       |                | retrain- |
| --------- | ---------- | ---------- | --- | ------------ | ---------- | ----------- | -------- | --------------------- | ------------ | -------------- | ------------ | --------- | --------------- | -------------- | -------- |
| Second,   | to account | for        | QoS | based on the | datacenter |             | type and |                       |              |                |              |           |                 |                |          |
|           |            |            |     |              |            |             |          | ing. Finally,         | all          | neural network |              | weights   | are initialized |                | using    |
| the job’s | latency    | tolerance, | we  | incorporate  | a          | QoS reward, |          |                       |              |                |              |           |                 |                |          |
|           |            |            |     |              |            |             |          | Xavier initialization |              | [43],          | with         | biases    | set to          | zero, ensuring |          |
|           |            |            |     | QoS          |            |             |          | efficient             | learning     | and stable     | convergence. |           |                 |                |          |
|           |            |            | R = |              | ,          |             |          |                       |              |                |              |           |                 |                |          |
q jobsPlaced
with quality values as detailed in Table I. Third, to penalize V. MODELEXTENSIONS
deadline violations, we compute a deadline violation ratio, VI. PERFORMANCEEVALUATION
deadlineViolationsCount Experimental Setup. We evaluate our approach using the
|     |     | R = |             |     |     | .   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | d   | jobsWaiting |     |     |     |     |     |     |     |     |     |     |     |     |
CloudSimPlussimulator[44],andacustomDRLenvironment
The overall reward is given by built on the Gymnasium API [45]. To bridge the Java-based
|     |     |     |      |      |     |     |     | CloudSim | Plus | with Python | DRL | agents, | we  | used the | Py4J |
| --- | --- | --- | ---- | ---- | --- | --- | --- | -------- | ---- | ----------- | --- | ------- | --- | -------- | ---- |
|     |     | R=c | R +c | R −c | R , |     |     |          |      |             |     |         |     |          |      |
p p q q d d gateway [46], adding the functionality to dynamically support
|         |               |     |                  |       |     |            |        | job-to-datacenter |     | placement | decisions. |     |     |     |     |
| ------- | ------------- | --- | ---------------- | ----- | --- | ---------- | ------ | ----------------- | --- | --------- | ---------- | --- | --- | --- | --- |
| where c | p , c q , and | c d | are coefficients | (with | 0   | ≤ c p ,c q | ,c d ≤ | 1                 |     |           |            |     |     |     |     |
and c +c +c = 1) that adjust the relative importance of For the calculation of rewards, we assign equal weights
| p   | q   | d   |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
each component. to placement, quality of service, and deadline compliance
Policy Network Architecture. With increasing system com- metrics (coefficients c p = c q = c d = 0.33). The experiments
|           |           |      |           |       |       |                  |         | used the      | PPO  | algorithm     | from | the stable-baselines3 |          |          | (SB3) |
| --------- | --------- | ---- | --------- | ----- | ----- | ---------------- | ------- | ------------- | ---- | ------------- | ---- | --------------------- | -------- | -------- | ----- |
| plexity,  | designing | DRL  | solutions | often | leads | to incorporating |         |               |      |               |      |                       |          |          |       |
|           |           |      |           |       |       |                  |         | library [47], | with | two variants: |      | (i) the               | baseline | PPO-Base |       |
| ever more | features  | into | the state | space | in an | effort to        | capture |               |      |               |      |                       |          |          |       |
everynuanceoftheinfrastructure.However,thisexpansionhas (vanilla SB3 implementation), and (ii) our extended PPO-X,
|           |            |     |        |                   |     |                 |     | which integrates |     | a custom | feature | extractor | (see | Figure | 3) to |
| --------- | ---------- | --- | ------ | ----------------- | --- | --------------- | --- | ---------------- | --- | -------- | ------- | --------- | ---- | ------ | ----- |
| two major | drawbacks. |     | First, | it slows training |     | and convergence |     |                  |     |          |         |           |      |        |       |
due to the curse of dimensionality phenomenon [42]. Second, enhance adaptability to infrastructure changes. Training was
|     |     |     |     |     |     |     |     | conducted | on NVIDIA | RTX | 4080/4090 |     | GPUs | with Intel | i7- |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | --------- | --- | --------- | --- | ---- | ---------- | --- |
ittightlycouplestheagenttoaspecificinfrastructure,forcinga
14700KF/i9-14900KFCPUsover600ktimesteps(about5to7
completeredesignofthestatespace,actionspace,andreward
functionwhentransitioningtoanewenvironment—evenwhen hours per run), with a simulation timestep of 1 second. Each
|     |     |     |     |     |     |     |     | experiment | was | repeated | using | five different |     | random | seeds, |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | --- | -------- | ----- | -------------- | --- | ------ | ------ |
theunderlyingtasksareconceptuallysimilar.Althoughfeature
|             |          |         |                |                |          |            |        | with the        | final results | averaged    |     | across these | runs.     |     |         |
| ----------- | -------- | ------- | -------------- | -------------- | -------- | ---------- | ------ | --------------- | ------------- | ----------- | --- | ------------ | --------- | --- | ------- |
| importance  | analysis |         | helps identify | which          | features | contribute |        |                 |               |             |     |              |           |     |         |
|             |          |         |                |                |          |            |        | Test Scenarios. |               | We compared |     | our PPO-X    | algorithm |     | against |
| positively, | these    | methods | remain         | time-consuming |          | and        | do not |                 |               |             |     |              |           |     |         |
fully decouple the agent from environment-specific details. two baselines, a heuristic and a vanilla DRL algorithm:
Moreover, new infrastructures may introduce additional fea- 1) Heuristic:Prioritizesjobsaccordingtotheirdeadlineand
tures,andthosepreviouslyconsideredimportantmightchange, criticality, placing them in the closest available datacenter.
furtherexacerbatingthestatedimensionmismatchissueinher- To enhance its effectiveness, we intentionally designed it to
ent in cross-domain transfers. allowmultipleplacementattemptsperjob,unlikeDRLagents,
To address these challenges, we adopt a more efficient whichmakeasingledecisionperstep.Thisgivestheheuristic
technique based on state abstraction. Our DRL agent incorpo- an advantage, as repeated placement attempts increase the
rates a custom feature extractor within its policy network that chances of finding available resources.

ASLANIDISetal.:CROSS-DOMAINDRLAGENTSFOREFFICIENTJOBPLACEMENTINTHECLOUD-EDGECONTINUUM 7
Train Env B Transfer Env B → Env A Transfer Env B → C
15.5
environment 12 10
0 15.0 10
raw observation API −10 14.5
8 −20
6 −30 14.0
feature padding 4 −40 13.5
0 300k 600k 0 300k 600k 0 300k 600k
Steps
agent
actor network critic
network
feature separator
fully
connected
categorical continuous
reward
features features
ReLu
embedding
normalization
MLP MLP state value
dropout +
fully adaptation
connected (residual)
fully advantage
ReLu
connected calculation
action distribution
action advantage
selection
action vector
Fig. 3. DRL agent architecture for handling state-action mismatches in
transfersbetweendifferentenvironments.
2) PPO-Base:ThevanillaPPOprovidedbytheSB3library.
Notethatobservationsarealwayszero-paddedbeforebeing
passed to the agent, ensuring a consistent state dimension
and allowing even the vanilla PPO algorithm to handle them
without mismatches.
The algorithms were trained in the topology of Figure 1.
We created two additional environments for transfer learning
experiments.TheenvironmentAremovestheclouddatacenter
from B to test the adaptability of the agent to a reduced
infrastructure. The environment C extends B with two new
datacenters: an edge datacenter in Copenhagen and a far-
edge datacenter at the AAU CPH university, both connected
to the Rotterman cloud datacenter. The jobs in C arrive at
three far-edge locations instead of two in B, and the job
traceisdifferent,withunseenjobdescriptions.Thedatacenter
characteristicsareasfollows:16hostswith64CPUcoreseach
for the cloud datacenter; 8 hosts with 16 CPU cores each for
the edge datacenters; 2-3 hosts with 6 CPU cores each for
the far-edge datacenters. All cores have the same processing
speed. Synthetic datasets were generated with job durations
uniformly distributed between 3−5 seconds, requested CPU
cores ranging from 1 to 20, soft deadlines between 0 − 5
seconds, and arrival rate of 1−4 jobs per timestep, totaling
50 jobs per episode.
Observations. In Figure 4, we see that the converged rewards
draweR
PPO-X PPO-Base Heuristic
(a) (b) (c)
Fig.4. Rewardcomparisonofourmethodvs.baselines.
Train Env B Transfer Env B → A Transfer Env B → C
1.0 1.0 1.0
0.8 0.8 0.8
0.6 0.6 0.6
0.4 0.4 0.4
0.2 0.2 0.2
0.0 0.0 0.0
Jobs QoS Deadlines Jobs QoS Deadlines Jobs QoS Deadlines
PlacedLatencyViolated PlacedLatencyViolated PlacedLatencyViolated
Ratio Ratio Ratio Ratio Ratio Ratio Ratio Ratio Ratio
Performance Metrics
oitaR
ecnamrofreP
PPO-X PPO-Base Heuristic
(a) (b) (c)
Fig.5. Performancemetrics:ourmethodvs.baselines.
of both PPO variants (at 600k steps) exceed the heuristic’s
performance in all three environments. We also observe that
during training, our PPO extension consistently outperforms
the standard PPO algorithm (Figure 4a). However, it is im-
portant to note that our extension is not always necessary
and may converge more slowly than the simpler version of
PPO (Figure 4b). For example, in environments such as A,
wherethestate-actionspaceisrelativelysmallandtheoptimal
solution is easier to identify, our extension may not provide
significant benefits. Due to the larger model size, adaptation
canbeslowerasmoreparametersneedtobeupdated.Insuch
cases, it is crucial to evaluate whether the added complexity
ofourextensionjustifiesitsuseoverthesimplerPPOversion.
At the same time, in Figure 5, we see how total rewards
translate into practical system performance. Specifically, the
jobs placed and QoS latency ratios should be maximized,
while the deadlines violated ratio should be minimized. We
observe that the performance metrics in environment A are
actually worse compared to environment B, which may seem
counterintuitive given that environment A is less complex
and thus presumably easier to solve. The reason for the
bad performance is that while environment A has a much
smaller state-action space, making it computationally simpler
to solve, it also has far fewer hosts available. As a result, the
workload pattern becomes highly stressful, leading to worse
job placement and deadline violation ratios compared to the
environment B, which is more resource rich.
The advantage of our extension is evident in Figure 4c,
where the environment presents greater challenges compared
to what the agent encountered during training. To better
understand the impact of workload variability, we designed a
test with a different workload trace, evaluating how both PPO
architecturesrespondtosuddenandcriticalchanges.Asshown
in Figure 4c, our extension not only converges to a superior
overallsolution,butalsoexhibitsgreaterstability,especiallyin

8 IEEETRANSACTIONSONCLOUDCOMPUTING,VOL.XX,NO.X,2026
the second half of the run, whereas the PPO-base algorithm patternchanges.Thisreducestheneedforextensiveretraining,
remains highly unstable, as indicated by the shaded regions simplifying real-world deployment.
representing variability across different seeds. This stability Moving forward, we aim to evaluate our approach in real-
suggests that PPO-X is better suited for environments where world cloud infrastructures, focusing on performance, cost
conditions fluctuate significantly, making it a more robust efficiency, and energy savings. A key goal is quantifying the
choice. benefitsofreusingagentknowledgeacrossdifferentinfrastruc-
tures.Wealsoplantoexploremulti-agentandhierarchicalRL,
VII. DISCUSSION where specialized agents manage datacenter selection, host
allocation, VM scaling, and job migration, enabling scalable,
A. Scalability and Telemetry Overhead Analysis
decentralized decision-making. To enhance resilience, we will
Here,discussobservations,insights,limitationsandhow
investigate drift detection and adaptation by introducing host
it can fit to the real world.
failures or workload surges and ensuring agents can detect
Inarealisticcloud-edgedeployment,acrucialconsideration
and quickly respond to drift. Additionally, online learning
for reinforcement learning-based resource allocation is the
and adaptive reward shaping will refine policies in real time,
volume and timeliness of telemetry data required by the
improving stability and convergence. Another direction is
agent to make decisions effectively. This section provides
integrating uncertainty-aware RL and contrastive learning to
a first-order estimation of the telemetry data rate and the
improvedecision-makinginunpredictableenvironments.Con-
correspondingbandwidthandlatencyconstraints,asafunction
tinualandmeta-learningcouldenhancelong-termadaptability
of the scale of the infrastructure.
while preventing catastrophic forgetting. Finally, using graph
Telemetry Volume Estimation: Let D be the number of data-
neural networks in the policy network could better capture
centers, and let N denote the number of nodes (e.g., hosts or
d relationshipsbetweencloud-edgeresources,improvingrobust-
edge devices) in datacenter d ∈ {1,...,D}. Assume that the
ness in large-scale deployments.
agent receives a fixed-size telemetry vector s ∈Rk per node,
i
representing the state features (e.g., CPU utilization, memory REFERENCES
usage, power state, etc.).
[1] B. P. Rimal, E. Choi, and I. Lumb, “A taxonomy and survey of cloud
The total telemetry volume V per timestep is then:
tele computing systems,” in 2009 Fifth International Joint Conference on
INC,IMSandIDC,2009,pp.44–51.
D
(cid:88) [2] A. I. Al-Fuqaha, M. Guizani, M. Mohammadi, M. Aledhari, and
V = N ·k·s (in bytes)
tele d f M. Ayyash, “Internet of things: A survey on enabling technologies,
d=1 protocols, and applications,” IEEE Commun. Surv. Tutorials, vol. 17,
no.4,pp.2347–2376,2015.
wheres isthesizeinbytesofeachfeature(typically4bytes
f [3] M. Satyanarayanan, “The emergence of edge computing,” Computer,
for 32-bit floats). vol.50,no.1,pp.30–39,2017.
Bandwidth Requirement. If the agent must receive telemetry [4] D. Rosendo, A. Costan, P. Valduriez, and G. Antoniu, “Distributed
intelligence on the edge-to-cloud continuum: A systematic literature
data every ∆t seconds to match the job arrival rate and make
review,”J.ParallelDistributedComput.,vol.166,pp.71–94,2022.
timely decisions, the required bandwidth B becomes: [5] T. Aslanidis, A. Chouliaras, and D. Chatzopoulos, “Reinforcement
learningtechniquesforoptimizingsystemconfigurationonthecloud:A
B = V tele (in bytes/second) taxonomyandopenproblems,”inProceedingsofthe2023International
∆t ConferenceonembeddedWirelessSystemsandNetworks,EWSN2023,
2023,pp.345–350.
Latency Consideration: The end-to-end latency L end for [6] N.Liu,Z.Li,J.Xu,Z.Xu,S.Lin,Q.Qiu,J.Tang,andY.Wang,“A
telemetry delivery (sensor → aggregation → agent) must be hierarchicalframeworkofcloudresourceallocationandpowermanage-
significantly lower than ∆t, ideally L < 0.25 · ∆t, to ment using deep reinforcement learning,” in 37th IEEE International
end ConferenceonDistributedComputingSystems,ICDCS2017,2017,pp.
ensure that decisions are made on fresh state information 372–382.
and avoid queue buildup. Otherwise, the system risks acting [7] Z. Yang, P. Nguyen, H. Jin, and K. Nahrstedt, “MIRAS: model-based
reinforcementlearningformicroserviceresourceallocationoverscien-
on outdated states, potentially increasing job wait times or
tificworkflows,”in39thIEEEInternationalConferenceonDistributed
resource underutilization. ComputingSystems,ICDCS2019,2019,pp.122–132.
Implications. This analysis emphasizes the need for efficient [8] Y.Ran,H.Hu,X.Zhou,andY.Wen,“Deepee:Jointoptimizationofjob
scheduling and cooling control for data center energy efficiency using
telemetry aggregation and lightweight representations (e.g.,
deepreinforcementlearning,”in39thIEEEInternationalConferenceon
GNN embeddings, sketching) when scaling to larger infras- DistributedComputingSystems,ICDCS2019,2019,pp.645–655.
tructures.Infutureextensions,weplantoexplorethetrade-off [9] J. Zeng, D. Ding, K. Kang, H. Xie, and Q. Yin, “Adaptive drl-based
virtual machine consolidation in energy-efficient cloud data center,”
between state fidelity and communication overhead, including
IEEETrans.ParallelDistributedSyst.,vol.33,no.11,pp.2991–3002,
decentralized agents or hierarchical learning architectures. 2022.
[10] M.Ferens,D.Hortelano,I.DeMiguel,R.J.D.Barroso,andS.Kosta,
“Sterocen:Simulationandtrainingenvironmentforresourceorchestra-
VIII. CONCLUSIONS&FUTUREWORK
tionincloud-edgenetworks,”in202415thInternationalConferenceon
We present a novel approach to enhance the adaptability NetworkoftheFuture(NoF). IEEE,2024,pp.133–141.
[11] Y.Zhang,B.Di,Z.Zheng,J.Lin,andL.Song,“Distributedmulti-cloud
and transferability of DRL agents in dynamic cloud-edge en-
multi-access edge computing by multi-agent reinforcement learning,”
vironments. Our architecture minimizes infrastructure-specific IEEETrans.Wirel.Commun.,vol.20,no.4,pp.2565–2578,2021.
dependencies, enabling cross-domain transfers. By learning [12] B. Kruekaew and W. Kimpan, “Multi-objective task scheduling op-
timization for load balancing in cloud computing environment using
infrastructure-agnostic state representations, DRL agents can
hybrid artificial bee colony algorithm with reinforcement learning,”
generalizeeffectivelydespiteresourceavailabilityorworkload IEEEAccess,vol.10,pp.17803–17818,2022.

ASLANIDISetal.:CROSS-DOMAINDRLAGENTSFOREFFICIENTJOBPLACEMENTINTHECLOUD-EDGECONTINUUM 9
[13] M. Hu, H. Wang, X. Xu, J. He, Y. Hu, T. Deng, and K. Peng, “Joint [34] J. Xing, T. Nagata, K. Chen, X. Zou, E. Neftci, and J. L. Krichmar,
optimizationofmicroservicedeploymentandroutinginedgeviamulti- “Domain adaptation in reinforcement learning via latent unified state
objectivedeepreinforcementlearning,”IEEETrans.Netw.Serv.Manag., representation,” in Thirty-Fifth AAAI Conference on Artificial Intelli-
vol.21,no.6,pp.6364–6381,2024. gence,AAAI2021,2021,pp.10452–10459.
[14] C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning [35] N. Botteghi, M. Poel, and C. Brune, “Unsupervised representation
for fast adaptation of deep networks,” in Proceedings of the 34th learningindeepreinforcementlearning:Areview,”2022.
International Conference on Machine Learning, ICML 2017, vol. 70, [36] T.Zeng,X.Zhang,J.Duan,C.Yu,C.Wu,andX.Chen,“Anoffline-
2017,pp.1126–1135. transfer-online framework for cloud-edge collaborative distributed re-
[15] K. Khetarpal, M. Riemer, I. Rish, and D. Precup, “Towards continual inforcement learning,” IEEE Trans. Parallel Distributed Syst., vol. 35,
reinforcementlearning:Areviewandperspectives,”J.Artif.Intell.Res., no.5,pp.720–731,2024.
vol.75,pp.1401–1476,2022. [37] K. Shuai, Y. Miao, K. Hwang, and Z. Li, “Transfer reinforcement
[16] H. Qiu, W. Mao, C. Wang, H. Franke, A. Youssef, Z. T. Kalbarczyk, learningforadaptivetaskoffloadingoverdistributededgeclouds,”IEEE
T.Basar,andR.K.Iyer,“AWARE:automateworkloadautoscalingwith Trans.CloudComput.,vol.11,no.2,pp.2175–2187,2023.
reinforcementlearninginproductioncloudsystems,”inProceedingsof [38] Y. Tao, J. Qiu, and S. Lai, “A hybrid cloud and edge control strategy
the 2023 USENIX Annual Technical Conference, USENIX ATC 2023, for demand responses using deep reinforcement learning and transfer
2023,pp.387–402. learning,”IEEETrans.CloudComput.,vol.10,no.1,pp.56–71,2022.
[17] G.I.Parisi,R.Kemker,J.L.Part,C.Kanan,andS.Wermter,“Continual [39] Y. Bengio, R. Ducharme, P. Vincent, and C. Janvin, “A neural proba-
lifelonglearningwithneuralnetworks:Areview,”NeuralNetworks,vol. bilisticlanguagemodel,”J.Mach.Learn.Res.,vol.3,pp.1137–1155,
| 113,pp.54–71,2019. |     |     |     |     |     |     | 2003. |
| ------------------ | --- | --- | --- | --- | --- | --- | ----- |
[18] M.D.Lange,R.Aljundi,M.Masana,S.Parisot,X.Jia,A.Leonardis, [40] T.Mikolov,K.Chen,G.Corrado,andJ.Dean,“Efficientestimationof
1st International Conference
G.G.Slabaugh,andT.Tuytelaars,“Acontinuallearningsurvey:Defying word representations in vector space,” in
forgetting in classification tasks,” IEEE Trans. Pattern Anal. Mach. onLearningRepresentations,ICLR2013,WorkshopTrackProceedings,
| Intell.,vol.44,no.7,pp.3366–3385,2022. |     |     |     |     |     |     | 2013. |
| -------------------------------------- | --- | --- | --- | --- | --- | --- | ----- |
[19] J.Lu,A.Liu,F.Dong,F.Gu,J.Gama,andG.Zhang,“Learningunder [41] K.He,X.Zhang,S.Ren,andJ.Sun,“Deepresiduallearningforimage
conceptdrift:Areview,”IEEETrans.Knowl.DataEng.,vol.31,no.12, recognition,”in2016IEEEConferenceonComputerVisionandPattern
| pp.2346–2363,2019. |     |     |     |     |     |     | Recognition,CVPR2016,2016,pp.770–778. |
| ------------------ | --- | --- | --- | --- | --- | --- | ------------------------------------- |
[20] J. Li, C. Hsu, M. Chang, and W. Chen, “A comprehensive review of [42] R.Bellman,DynamicProgramming. DoverPublications,1957.
machine learning advances on data change: A cross-field perspective,” [43] X. Glorot and Y. Bengio, “Understanding the difficulty of training
CoRR,vol.abs/2402.12627,2024. deep feedforward neural networks,” in Proceedings of the thirteenth
[21] M.Wan,T.Gangwani,andJ.Peng,“Mutualinformationbasedknowl- internationalconferenceonartificialintelligenceandstatistics. JMLR
edge transfer under state-action dimension mismatch,” in Proceedings WorkshopandConferenceProceedings,2010,pp.249–256.
oftheThirty-SixthConferenceonUncertaintyinArtificialIntelligence, [44] M. C. Silva Filho, R. L. Oliveira, C. C. Monteiro, P. R. Ina´cio, and
UAI2020,vol.124,2020,pp.1218–1227. M.M.Freire,“Cloudsimplus:acloudcomputingsimulationframework
[22] X. Wen, C. Bai, K. Xu, X. Yu, Y. Zhang, X. Li, and Z. Wang, pursuing software engineering principles for improved modularity, ex-
“Contrastiverepresentationfordatafilteringincross-domainofflinerein- tensibilityandcorrectness,”in2017IFIP/IEEEsymposiumonintegrated
forcementlearning,”inForty-firstInternationalConferenceonMachine networkandservicemanagement(IM). IEEE,2017,pp.400–406.
Learning,ICML2024,2024. [45] M.Towers,A.Kwiatkowski,J.Terry,J.U.Balis,G.DeCola,T.Deleu,
[23] J.Lyu,C.Bai,J.Yang,Z.Lu,andX.Li,“Cross-domainpolicyadapta- M. Goula˜o, A. Kallinteris, M. Krimmel, A. KG et al., “Gymnasium:
tion by capturing representation mismatch,” in Forty-first International A standard interface for reinforcement learning environments,” arXiv
ConferenceonMachineLearning,ICML2024,2024. preprintarXiv:2407.17032,2024.
[24] T. Yang, H. You, J. Hao, Y. Zheng, and M. E. Taylor, “A transfer [46] W. Funika, P. Koperek, and J. Kitowski, “Repeatable experiments in
approachusinggraphneuralnetworksindeepreinforcementlearning,” the cloud resources management domain with use of reinforcement
inThirty-EighthAAAIConferenceonArtificialIntelligence,AAAI2024, learning,”inCracowGridWorkshop,2018,pp.31–32.
2024,pp.16352–16360. [47] A.Raffin,A.Hill,A.Gleave,A.Kanervisto,M.Ernestus,andN.Dor-
[25] Y.Chen,Y.Chen,Y.Yang,Y.Li,J.Yin,andC.Fan,“Learningaction- mann,“Stable-baselines3:Reliablereinforcementlearningimplementa-
transferablepolicywithactionembedding,”CoRR,vol.abs/1909.02291, tions,”JournalofMachineLearningResearch,vol.22,no.268,pp.1–8,
| 2019. |     |     |     |     |     |     | 2021. |
| ----- | --- | --- | --- | --- | --- | --- | ----- |
[26] S.A.Serrano,J.Mart´ınez-Carranza,andL.E.Sucar,“Similarity-based
| knowledge | transfer | for | cross-domain | reinforcement | learning,” |     | CoRR, |
| --------- | -------- | --- | ------------ | ------------- | ---------- | --- | ----- |
vol.abs/2312.03764,2023.
| [27] E. Todorov, | T.        | Erez, and | Y. Tassa,     | “Mujoco:      | A physics | engine     | for |
| ---------------- | --------- | --------- | ------------- | ------------- | --------- | ---------- | --- |
| model-based      | control,” | in        | 2012 IEEE/RSJ | International |           | Conference | on  |
IntelligentRobotsandSystems,IROS2012,2012,pp.5026–5033.
| [28] G. Joshi | and G. | Chowdhary, | “Cross-domain |     | transfer in | reinforcement |     |
| ------------- | ------ | ---------- | ------------- | --- | ----------- | ------------- | --- |
learningusingtargetapprentice,”in2018IEEEInternationalConference
onRoboticsandAutomation,ICRA2018,2018,pp.7525–7532.
| [29] H. You, | T. Yang, | Y. Zheng,     | J. Hao, | and M.         | E. Taylor,      | “Cross-domain |        |
| ------------ | -------- | ------------- | ------- | -------------- | --------------- | ------------- | ------ |
| adaptive     | transfer | reinforcement |         | learning based | on state-action |               | corre- |
spondence,”inUncertaintyinArtificialIntelligence,Proceedingsofthe
Thirty-EighthConferenceonUncertaintyinArtificialIntelligence,UAI
2022,vol.180,2022,pp.2299–2309.
[30] H.Bou-Ammar,E.Eaton,P.Ruvolo,andM.E.Taylor,“Unsupervised
cross-domaintransferinpolicygradientreinforcementlearningviaman-
ifoldalignment,”inProceedingsoftheTwenty-NinthAAAIConference
| onArtificialIntelligence. |            |              | AAAIPress,2015,pp.2504–2510. |              |                |           |          |
| ------------------------- | ---------- | ------------ | ---------------------------- | ------------ | -------------- | --------- | -------- |
| [31] I. Higgins,          | A.         | Pal, A. A.   | Rusu,                        | L. Matthey,  | C. P. Burgess, | A.        | Pritzel, |
| M. M.                     | Botvinick, | C. Blundell, | and                          | A. Lerchner, | “DARLA:        | improving |          |
zero-shottransferinreinforcementlearning,”inProceedingsofthe34th
| International | Conference |     | on Machine | Learning, | ICML | 2017, | vol. 70, |
| ------------- | ---------- | --- | ---------- | --------- | ---- | ----- | -------- |
2017,pp.1480–1490.
| [32] X. Wang, | H.  | Chen, Y. | Zhou, | J. Ma, and | W. Zhu, | “Disentangled |     |
| ------------- | --- | -------- | ----- | ---------- | ------- | ------------- | --- |
representationlearningforrecommendation,”IEEETrans.PatternAnal.
Mach.Intell.,vol.45,no.1,pp.408–424,2023.
[33] M.Laskin,A.Srinivas,andP.Abbeel,“CURL:contrastiveunsupervised
representationsforreinforcementlearning,”inProceedingsofthe37th
| International | Conference |     | on Machine | Learning, | ICML | 2020, vol. | 119, |
| ------------- | ---------- | --- | ---------- | --------- | ---- | ---------- | ---- |
2020,pp.5639–5650.

10 IEEETRANSACTIONSONCLOUDCOMPUTING,VOL.XX,NO.X,2026
TheodorosAslanidisisaPhDStudentintheSchool
ofComputerScienceofUniversityCollegeDublin.
HereceivedhisDiplomainElectricalandComputer
Engineering from the University of Thessaly. On
his PhD, he works on the MLSysOps EU project,
focusingonAIforautonomicsystemoperation.His
researchinterestsincludecloudcomputing,resource
orchestration,andmachinelearningforsystems.
Sokol Kosta is associate professor at the Dept. of
ElectronicSystems,AalborgUniversity.Hisresearch
includesNetworking,DistributedSystems,Security,
andEdgeComputing.BSc,MSc,PhD,andPostDoc
in Computer Science from Sapienza University of
Rome and a visiting researcher with HKUST in
2015. He has won the Best PhD Student Paper
award by the Computer Science Dept. of Sapienza
University (2012), the IEEE INFOCOM and IEEE
SECON Best Demo awards (2013), and the IEEE
INFOCOMTestofTimepaperaward(2024).
Spyros Lalis is Professor at the ECE Depart-
ment, University of Thessaly, Greece. He received
a Diploma in Computer Science and a Ph.D. in
Technical Sciences from ETHZ, and has worked
attheComputerScienceDepartment,Universityof
Crete,theFoundationforResearchandTechnology
Hellas (FORTH) and the Center for Research and
TechnologyHellas(CERTH).Hisresearchinterests
are mainly in programming models, operating sys-
tems,distributedsystemsandubiquitouscomputing.
He has actively contributed in the development of
system software and middleware for market-based resource allocation, dis-
tributedcomputingandmetacomputing,ad-hocwearablecomputing,wireless
sensor/actuatornetworks,mobile/crowdsensinganddronebasedsystems.He
haspublishedover100scientificpapersininternationaljournalsandconfer-
ences,andhasreceivedsignificantfundingforhisworkthroughcompetitive
EUandnationalresearchprojects.
Dimitris Chatzopoulos received the diploma and
MScdegreesincomputerengineeringandcommu-
nications from the University of Thessaly, Greece,
and the PhD degree in computer science and engi-
neeringfromtheHongKongUniversityofScience
and Technology. He is an assistant professor with
the School of Computer Science, University Col-
legeDublin. Hisresearch interestsinclude privacy-
preserving and AI-enabled decentralized applica-
tionsformobileanddistributedsystems.
