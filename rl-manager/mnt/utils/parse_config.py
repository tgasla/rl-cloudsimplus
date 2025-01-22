import yaml


class Datacenter:
    def __init__(self, name, dc_type, amount, hosts=None):
        self.name = name
        self.dc_type = dc_type
        self.amount = amount
        self.hosts = hosts if isinstance(hosts, list) else [hosts] if hosts else []

    def __repr__(self):
        return f"Datacenter(name={self.name}, type={self.dc_type}, amount={self.amount}, hosts={self.hosts})"

    def to_dict(self):
        return {
            "name": self.name,
            "type": self.dc_type,
            "amount": self.amount,
            "hosts": [host.to_dict() for host in self.hosts],
        }


class Host:
    def __init__(self, amount, pes, pe_mips, ram, storage, bw, vms=None):
        self.amount = amount
        self.pes = pes
        self.pe_mips = pe_mips
        self.ram = ram
        self.storage = storage
        self.bw = bw
        self.vms = vms if isinstance(vms, list) else [vms] if vms else []

    def __repr__(self):
        return f"Host(amount={self.amount}, pes={self.pes}, pe_mips={self.pe_mips}, ram={self.ram}, storage={self.storage}, bw={self.bw}, vms={self.vms})"

    def to_dict(self):
        return {
            "amount": self.amount,
            "pes": self.pes,
            "pe_mips": self.pe_mips,
            "ram": self.ram,
            "storage": self.storage,
            "bw": self.bw,
            "vms": [vm.to_dict() for vm in self.vms],
        }


class Vm:
    def __init__(self, amount, pes, pe_mips, ram, size, bw):
        self.amount = amount
        self.pes = pes
        self.pe_mips = pe_mips
        self.ram = ram
        self.size = size
        self.bw = bw

    def __repr__(self):
        return f"Vm(amount={self.amount}, pes={self.pes}, pe_mips={self.pe_mips}, ram={self.ram}, size={self.size}, bw={self.bw})"

    def to_dict(self):
        return {
            "amount": self.amount,
            "pes": self.pes,
            "pe_mips": self.pe_mips,
            "ram": self.ram,
            "size": self.size,
            "bw": self.bw,
        }


# Custom constructors for YAML tags
def datacenter_constructor(loader, node):
    fields = loader.construct_mapping(node)
    return Datacenter(
        name=fields["name"],
        dc_type=fields["type"],
        amount=fields["amount"],
        hosts=fields.get("hosts", []),
    )


def host_constructor(loader, node):
    fields = loader.construct_mapping(node)
    return Host(
        amount=fields["amount"],
        pes=fields["pes"],
        pe_mips=fields["pe_mips"],
        ram=fields["ram"],
        storage=fields["storage"],
        bw=fields["bw"],
        vms=fields.get("vms", []),
    )


def vm_constructor(loader, node):
    fields = loader.construct_mapping(node)
    return Vm(
        amount=fields["amount"],
        pes=fields["pes"],
        pe_mips=fields["pe_mips"],
        ram=fields["ram"],
        size=fields["size"],
        bw=fields["bw"],
    )


def register_cnstructors():
    yaml.add_constructor("!datacenter", datacenter_constructor)
    yaml.add_constructor("!host", host_constructor)
    yaml.add_constructor("!vm", vm_constructor)


def dict_from_config(replica_id, config):
    register_cnstructors()
    with open(config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # first we read the common parameters and we overwrite them with the specific experiment parameters
    params = {**config["common"], **config[f"experiment_{replica_id}"]}
    # print(params["datacenters"])
    return params
