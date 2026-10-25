#!/usr/bin/env python3
"""Check editor C# and Unity APIs without starting Unity.

Project collaborators and HDRP are compile-only stand-ins. This check does not
validate scene construction or HDRP rendering.
"""
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
UNITY = Path('/Applications/Unity/Hub/Editor/6000.6.6f1/Unity.app/Contents/Resources')
SOURCES = ('World/FloorPlan.cs', 'World/Scenario.cs', 'Editor/HouseMenu.cs',
           'Editor/RecorderMenu.cs', 'Editor/RanchExteriorBuilder.cs')


def main():
    managed = UNITY / 'Scripting/Managed'
    sdk = UNITY / 'Scripting/DotNetSdk'
    refs = [managed / 'UnityEditor.dll', *sorted((managed / 'UnityEngine').glob('*.dll')),
            *sorted((managed / 'UnityEditor').glob('*.dll')), sdk / 'sdk/8.0.318/Newtonsoft.Json.dll']
    scratch = ROOT / 'tools/headless/obj'
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='ranch-editor-check-', dir=scratch) as temp:
        target = Path(temp)
        project = ET.Element('Project', Sdk='Microsoft.NET.Sdk')
        props = ET.SubElement(project, 'PropertyGroup')
        for key, value in dict(TargetFramework='net8.0', EnableDefaultCompileItems='false', NoWarn='CS0649').items():
            ET.SubElement(props, key).text = value
        items = ET.SubElement(project, 'ItemGroup')
        for source in SOURCES:
            ET.SubElement(items, 'Compile', Include=str(ROOT / 'unity/SmokeRescue/Assets/SmokeRescue' / source))
        ET.SubElement(items, 'Compile', Include='Stubs.cs')
        for i, path in enumerate(refs):
            ref = ET.SubElement(items, 'Reference', Include=f'ref{i}')
            ET.SubElement(ref, 'HintPath').text = str(path)
        (target / 'Stubs.cs').write_text((ROOT / 'tools/headless/RanchEditorStubs.txt').read_text())
        ET.ElementTree(project).write(target / 'EditorCheck.csproj', encoding='unicode')
        env = dict(os.environ, DOTNET_CLI_HOME='/private/tmp/headless-dotnet-home',
                   DOTNET_NOLOGO='1', DOTNET_CLI_TELEMETRY_OPTOUT='1')
        return subprocess.run([str(sdk / 'dotnet'), 'build', str(target / 'EditorCheck.csproj'),
                               '--ignore-failed-sources', '-v', 'q', '-nologo'], env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
