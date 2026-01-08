class Config:
    N: int = 64
    wavelength: float = 1.0
    na: float = 1.0
    dr: float = 1.0

    def __init__(self) -> None:
        self.dk = self.wavelength / 2 / self.na

    def __str__(self) -> str:
        output: str = "Project config: \n"
        for attribute_name in dir(self):
            value = getattr(self, attribute_name)
            if type(value) not in (int, float):
                continue
            output += f"{attribute_name}: {type(value).__name__} = {value}\n"
        return output


config = Config()
