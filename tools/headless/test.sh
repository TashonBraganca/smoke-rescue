#!/bin/sh
set -eu
here=$(cd "$(dirname "$0")" && pwd)
export SMOKE_REPO=$(cd "$here/../.." && pwd)
DOTNET=/Applications/Unity/Hub/Editor/6000.6.6f1/Unity.app/Contents/Resources/Scripting/DotNetSdk/dotnet
export DOTNET_CLI_HOME=/private/tmp/headless-dotnet-home DOTNET_NOLOGO=1 DOTNET_CLI_TELEMETRY_OPTOUT=1
"$DOTNET" build "$here/LogicTests.csproj" -c Release -o "$here/bin/tests" --ignore-failed-sources -v q -nologo
exec "$DOTNET" "$here/bin/tests/LogicTests.dll" "$@"
