#!/bin/sh
# Build the cloud engine (portable .NET 8 output in tools/ryor/engine/bin) from the current brain, fire and sensor sources.
set -e
here=$(cd "$(dirname "$0")" && pwd)
DOTNET=${DOTNET:-/Applications/Unity/Hub/Editor/6000.6.6f1/Unity.app/Contents/Resources/Scripting/DotNetSdk/dotnet}
export DOTNET_CLI_HOME=${DOTNET_CLI_HOME:-/private/tmp/headless-dotnet-home} DOTNET_NOLOGO=1 DOTNET_CLI_TELEMETRY_OPTOUT=1
python3 "$here/patch_headless.py" "$here/engine/gen/HeadlessSim.cs"
rm -rf "$here/engine/bin"
"$DOTNET" publish "$here/engine/RyorRunner.csproj" -c Release -o "$here/engine/bin" -p:UseAppHost=false -v q -nologo
ls -l "$here/engine/bin/RyorRunner.dll" "$here/engine/bin/Newtonsoft.Json.dll" "$here/engine/bin/UnityEngine.CoreModule.dll"
