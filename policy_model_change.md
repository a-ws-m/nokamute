I want to make a big change to the representation and architecture for the case
where we're using the policy model. We need an updated graph representation of
the board, specifically for this model.

The input representation should be a heterogeneous graph (see the
pytorch-geometric docs using Context7 to see how models + representations of
heterogeneous graphs work). Nodes for empty spaces around the board are called
"destination nodes", and we should also add destination nodes representing
accessible spaces on top of pieces (every piece will get an extra destination
node, unless they already have a piece on top of them). Nodes for pieces that
are currently on the board are called "in-play nodes".

Furthermore, we are going to have two types of edges: "neighbour edges",
representing nodes that are adjacent/on top of each other, and "move edges",
which are edges between destination nodes and piece nodes. Move edges represent
legal moves *for both players* (even if it's not their turn at the moment),
which we compute in rust. We are also going to include "out-of-play nodes" for
pieces not yet on the board: one out-of-play node for each type of piece still
available in the pool, and for each colour of said piece. These should be
connected by move edges to legal spaces where they are allowed be placed (again,
regardless of whose turn it currently is). They won't have any neighbour edges.

Additionally, we are going to change the features of the nodes and edges. We are
going to remove the height feature and the "current player" feature from all
nodes. Other than that, in-play nodes will keep the same features. Out-of-play
nodes will have the same features as in-play nodes. Destination nodes will have
a single meaningful feature: a binary value indicating whether the destination
is on top of the hive (1 if on top, 0 if on bottom), but it needs to be padded
with zeroes to match the size of the other node features. Move edges will have a
binary feature, too: this indicates whether the *corresponding move* is for the
current player (1) or for the next player (0). Neighbour edges don't need any
meaningful features, but they should have a single zero to match the size of the
move edge feature.

Next, we need to update the way the model actually works. Currently, we pool the
graph, then use an MLP to produce the action vector. Instead of this, we are
going to increase the default number of graph layers and have no pooling at all;
the action vector will be produced by reading the move edge feature vectors. To
do this, we need to pass an additional vector to the model input, providing a
mapping from move indexes to action vector indexes, to map each move edge with
initial feature 1 (indicating it's a legal move for the current player) to the
corresponding index in the action vector. I think this should be possible in
pure pytorch, without using a for loop (I want to make sure we don't get a graph
break here). For this to work, the move edge features should have a final
dimension of 1, being the logit/probit for that move.

Do testing as you go, and remember to `micromamba activate torch`. Also, make
sure this all works with the league training/testing code.
