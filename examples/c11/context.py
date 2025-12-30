class CNameContext:
    def __init__(self, *sets: set[str]) -> None:
        self.sets: list[set[str]] = list(sets) if sets else [set()]

    def __contains__(self, key: str, /) -> bool:
        return key in self.sets[-1]

    def declare_typedef_name(self, name: str, /) -> None:
        self.sets[-1].add(name)

    def declare_var_name(self, name: str, /) -> None:
        self.sets[-1].remove(name)

    def save_context(self, /) -> None:
        self.sets.append(set())

    def restore_context(self, /) -> None:
        self.sets.pop()
