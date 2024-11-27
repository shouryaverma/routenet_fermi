import tensorflow as tf

class RouteNet_Fermi(tf.keras.Model):
    def __init__(self):
        super(RouteNet_Fermi, self).__init__()

        # Configuration parameters
        self.max_num_models = 7
        self.num_policies = 4
        self.max_num_queues = 3
        self.iterations = 8
        self.path_state_dim = 32
        self.link_state_dim = 32
        self.queue_state_dim = 32

        # Z-score normalization parameters
        self.z_score = {
            'traffic': [1385.4059, 859.8119],
            'packets': [1.4015, 0.8933],
            'eq_lambda': [1350.9712, 858.3162],
            'avg_pkts_lambda': [0.9117, 0.9724],
            'exp_max_factor': [6.6636, 4.7151],
            'pkts_lambda_on': [0.9116, 1.6513],
            'avg_t_off': [1.6649, 2.3564],
            'avg_t_on': [1.6649, 2.3564],
            'ar_a': [0.0, 1.0],
            'sigma': [0.0, 1.0],
            'capacity': [27611.0918, 20090.6211],
            'queue_size': [30259.1055, 21410.0957]
        }

        # SimpleRNN Cells for message passing
        self.path_update = tf.keras.layers.SimpleRNNCell(self.path_state_dim)
        self.link_update = tf.keras.layers.SimpleRNNCell(self.link_state_dim)
        self.queue_update = tf.keras.layers.SimpleRNNCell(self.queue_state_dim)

        # Embedding layers
        self.path_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(10 + self.max_num_models,)),
            tf.keras.layers.Dense(self.path_state_dim, activation='relu'),
            tf.keras.layers.Dense(self.path_state_dim, activation='relu')
        ])

        self.queue_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.max_num_queues + 2,)),
            tf.keras.layers.Dense(self.queue_state_dim, activation='relu'),
            tf.keras.layers.Dense(self.queue_state_dim, activation='relu')
        ])

        self.link_embedding = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.num_policies + 1,)),
            tf.keras.layers.Dense(self.link_state_dim, activation='relu'),
            tf.keras.layers.Dense(self.link_state_dim, activation='relu')
        ])

        # Readout layer
        self.readout_path = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, self.path_state_dim)),
            tf.keras.layers.Dense(self.link_state_dim // 2, activation='relu'),
            tf.keras.layers.Dense(self.path_state_dim // 2, activation='relu'),
            tf.keras.layers.Dense(1)
        ], name="PathReadout")

    @tf.function
    def call(self, inputs):
        # Extract inputs
        traffic = inputs['traffic']
        packets = inputs['packets']
        length = inputs['length']
        model = inputs['model']
        eq_lambda = inputs['eq_lambda']
        avg_pkts_lambda = inputs['avg_pkts_lambda']
        exp_max_factor = inputs['exp_max_factor']
        pkts_lambda_on = inputs['pkts_lambda_on']
        avg_t_off = inputs['avg_t_off']
        avg_t_on = inputs['avg_t_on']
        ar_a = inputs['ar_a']
        sigma = inputs['sigma']
        capacity = inputs['capacity']
        policy = tf.one_hot(inputs['policy'], self.num_policies)
        queue_size = inputs['queue_size']
        priority = tf.one_hot(inputs['priority'], self.max_num_queues)
        weight = inputs['weight']
        queue_to_path = inputs['queue_to_path']
        link_to_path = inputs['link_to_path']
        path_to_link = inputs['path_to_link']
        path_to_queue = inputs['path_to_queue']
        queue_to_link = inputs['queue_to_link']

        # Compute load and packet size
        path_gather_traffic = tf.gather(traffic, path_to_link[:, :, 0])
        load = tf.reduce_sum(path_gather_traffic, axis=1) / capacity
        pkt_size = traffic / packets

        # Initialize states
        path_features = tf.concat([
            (traffic - self.z_score['traffic'][0]) / self.z_score['traffic'][1],
            (packets - self.z_score['packets'][0]) / self.z_score['packets'][1],
            tf.one_hot(model, self.max_num_models),
            (eq_lambda - self.z_score['eq_lambda'][0]) / self.z_score['eq_lambda'][1],
            (avg_pkts_lambda - self.z_score['avg_pkts_lambda'][0]) / self.z_score['avg_pkts_lambda'][1],
            (exp_max_factor - self.z_score['exp_max_factor'][0]) / self.z_score['exp_max_factor'][1],
            (pkts_lambda_on - self.z_score['pkts_lambda_on'][0]) / self.z_score['pkts_lambda_on'][1],
            (avg_t_off - self.z_score['avg_t_off'][0]) / self.z_score['avg_t_off'][1],
            (avg_t_on - self.z_score['avg_t_on'][0]) / self.z_score['avg_t_on'][1],
            (ar_a - self.z_score['ar_a'][0]) / self.z_score['ar_a'][1],
            (sigma - self.z_score['sigma'][0]) / self.z_score['sigma'][1]
        ], axis=1)
        path_state = self.path_embedding(path_features)

        link_features = tf.concat([load, policy], axis=1)
        link_state = self.link_embedding(link_features)

        queue_features = tf.concat([
            (queue_size - self.z_score['queue_size'][0]) / self.z_score['queue_size'][1],
            priority,
            weight
        ], axis=1)
        queue_state = self.queue_embedding(queue_features)

        # Message passing iterations
        for _ in range(self.iterations):
            # LINK AND QUEUE TO PATH
            queue_gather = tf.gather(queue_state, queue_to_path)
            link_gather = tf.gather(link_state, link_to_path)
            path_inputs = tf.concat([queue_gather, link_gather], axis=2)

            path_update_rnn = tf.keras.layers.RNN(
                self.path_update,
                return_sequences=True,
                return_state=True
            )
            previous_path_state = path_state
            path_state_sequence, path_state = path_update_rnn(
                path_inputs,
                initial_state=path_state
            )
            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence],
                axis=1
            )

            # PATH TO QUEUE
            path_gather = tf.gather_nd(path_state_sequence, path_to_queue)
            path_sum = tf.reduce_sum(path_gather, axis=1)
            queue_state, _ = self.queue_update(path_sum, [queue_state])

            # QUEUE TO LINK
            queue_gather = tf.gather(queue_state, queue_to_link)
            link_rnn = tf.keras.layers.RNN(
                self.link_update,
                return_sequences=False,
                return_state=False
            )
            link_state = link_rnn(queue_gather, initial_state=link_state)

        # Compute delays
        capacity_gather = tf.gather(capacity, link_to_path)
        input_tensor = path_state_sequence[:, 1:].to_tensor()
        occupancy_gather = self.readout_path(input_tensor)
        occupancy_gather = tf.RaggedTensor.from_tensor(
            occupancy_gather, lengths=length
        )
        queue_delay = tf.reduce_sum(occupancy_gather / capacity_gather, axis=1)
        trans_delay = pkt_size * tf.reduce_sum(1 / capacity_gather, axis=1)

        return queue_delay + trans_delay
