from pathlib import Path
from subprocess import check_call

import setuptools
from setuptools.command.install import install


class CustomInstallCommand(install):
    def run(self):
        # install前に実行するコマンドを書く
        print("api build")
        check_call(
            "cd " + str(Path(__file__).parent) + "/localenv;bash build_api.sh",
            shell=True,
        )
        print("api build finish")
        # install実行
        install.run(self)


setuptools.setup(
    name="localenv",
    version="2023.2.26",
    packages=["localenv"],
    package_data={"localenv": ["py.typed"]},
    cmdclass={
        "install": CustomInstallCommand,
    },
)
