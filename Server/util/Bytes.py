import struct


class Bytes:

    def __init__(self, arg_bytes=None):
        if arg_bytes is None:
            self.bytes = []
        else:
            self.bytes = [x for x in arg_bytes]

        self.read_index = 0

    def write(self, variable):
        self.write_bytes(variable.bytes)

    def write_byte(self, variable):
        self.bytes.append(variable)

    def read_byte(self):
        variable = self.bytes[self.read_index]
        self.read_index += 1
        return variable

    def write_bytes(self, variable):
        for v in variable:
            self.bytes.append(v)
        # self.bytes.append(x for x in variable)

    def read_bytes(self, length):
        variable = self.bytes[self.read_index:self.read_index + length]
        self.read_index += length
        return bytes(variable)

    def write_integer(self, variable):
        data = struct.pack("<i", variable)
        self.write_bytes(data)

    def read_integer(self):
        data = struct.unpack("<i", self.read_bytes(4))[0]
        return data

    def write_float(self, variable):
        data = struct.pack("<f", variable)
        self.write_bytes(data)

    def read_float(self):
        data = struct.unpack("<f", self.read_bytes(4))[0]
        return data

    def write_long(self, variable):
        data = struct.pack("<q", variable)
        self.write_bytes(data)

    def read_long(self):
        data = struct.unpack("<q", self.read_bytes(8))[0]
        return data

    def write_bool(self, variable):
        data = 1 if variable else 0
        self.write_byte(data)

    def read_bool(self):
        data = True if self.read_byte() == 1 else False
        return data

    def write_string(self, variable):
        data = variable.encode('utf-8')
        self.write_integer(len(data))
        self.write_bytes(data)

    def read_string(self):
        length = self.read_integer()
        data = self.read_bytes(length).decode('utf-8')
        return data

    # variable = (x, y, z)
    def write_vector(self, variable):
        self.write_float(variable[0])
        self.write_float(variable[1])
        self.write_float(variable[2])

    def read_vector(self):
        return tuple([self.read_float(), self.read_float(), self.read_float()])

    def read_list(self, length):
        data = []
        for i in range(length):
            data.append(self.read_byte())
        return data

    def write_map(self, map_data):
        width = len(map_data[0])
        height = len(map_data[1])
        self.write_byte(width)
        self.write_byte(height)

        for y in range(height):
            for x in range(width):
                self.write_float(map_data[y][x])

    def read_map(self):
        width = self.read_byte()
        height = self.read_byte()
        channel = self.read_byte()

        variable = []
        for y in range(height):
            y_list = []
            for x in range(width):
                x_list = []
                for c in range(channel):
                    x_list.append(self.read_byte())
                y_list.append(x_list)
            variable.append(y_list)
        return variable

    def read_float_map(self):
        width = self.read_byte()
        height = self.read_byte()
        channel = self.read_byte()

        variable = []
        for y in range(height):
            y_list = []
            for x in range(width):
                x_list = []
                for c in range(channel):
                    x_list.append(self.read_float())
                y_list.append(x_list)
            variable.append(y_list)
        return variable

    def read_trace_map(self):
        """
        channel = []
        for i in range(4):
            variable = []
            width = self.read_byte()
            height = self.read_byte()
            for y in range(height):
                y_list = []
                for x in range(width):
                    y_list.append(self.read_float())
                variable.append(y_list)
            channel.append(variable)
        """
        """
        state_list = []
        for i in range(1):
            variable = []
            width = self.read_byte()
            height = self.read_byte()
            channel = self.read_byte()
            for y in range(height):
                y_list = []
                for x in range(width):
                    x_list = []
                    for c in range(channel):
                        x_list.append(self.read_float())
                    y_list.append(x_list)
                variable.append(y_list)
            state_list.append(variable)
        """
        state_list = []
        for i in range(1):
            variable = []
            width = self.read_byte()
            height = self.read_byte()
            for y in range(height):
                y_list = []
                for x in range(width):
                    y_list.append(self.read_float())
                variable.append(y_list)
            state_list.append(variable)

        return state_list

    def read_state(self):
        # return [self.read_float(), self.read_float(), self.read_float()]
        # return [self.read_float(), self.read_float(), self.read_float(), self.read_map(), self.read_trace_map()]
        # return self.read_trace_map()
        return self.read_float_map()
