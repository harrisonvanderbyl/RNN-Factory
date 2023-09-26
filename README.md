# RNN Factory

Parrallelizable RNNs are a great way to speed up training and inference. However, creating such RNNs is not trivial. The modules that are often created for such a purpose need to be able to handle both singular tokens and sequences of tokens. This can be tricky as optimizations often lead to situations where it is a lot of work to convert between the two modes. This factory aims to simplify the process of creating such modules, and to make it easier to create new RNN architectures.

By implementing a subclass of the nn.Module class, the StateModule class is a virtual class that stores the state.
By default, the state is not used during training, but will automatically become active during inference.

Some recursive dictionary functions can be implemented to find all StateModules in the model, and to retrieve or set their state.

## Code

Code can be found at https://github.com/harrisonvanderbyl/RNN-Factory

You can find examples of RWK v5 in the `master` branch

