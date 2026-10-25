# Hosted models

[README](../README.md) · [CLI guide](usage.md) · [Catalogue and review](catalogue.md)

**Recommended hosted provider: Google Gemini.** Gemini is the team's tested hosted
option, with successful smoke tests of the hosted workflow. Start with a Gemini
vision model available to your account. These checks establish workflow compatibility;
transcriptions still require human review.

Local Ollama is the default. Hosted mode sends sanitised detection and crop PNG
pixels over HTTPS to a provider. Original files and EXIF are not sent. Obtain
written client approval before sending private material, and check the provider's
current data handling, model access, charges and quotas. Use invented images for
connection tests.

## Configure a provider

Use the recommended `gemini/` prefix. The `openai/` and `anthropic/` adapters are
also supported; the project's recorded live smoke-test evidence is for Gemini.
All use direct API origins. Choose an available model that accepts images and
returns text. Set `UMA_BCT_API_MODEL` to its provider-qualified name and
`UMA_BCT_API_KEY` to its key. Read the key interactively to avoid shell history.

For a prepared ZIP in Bash:

```bash
export UMA_BCT_API_MODEL=gemini/your-vision-model
printf 'API key: '
read -r -s UMA_BCT_API_KEY
printf '\n'
export UMA_BCT_API_KEY
sh run-uma-bct.sh
unset UMA_BCT_API_KEY UMA_BCT_API_MODEL
```

Replace the example model with one available to your account. In PowerShell:

```powershell
$env:UMA_BCT_API_MODEL = 'gemini/your-vision-model'
$secret = Read-Host 'API key' -AsSecureString
$env:UMA_BCT_API_KEY = [System.Net.NetworkCredential]::new('', $secret).Password
& '.\Run UMA BCT.cmd'
Remove-Item Env:UMA_BCT_API_KEY, Env:UMA_BCT_API_MODEL
Remove-Variable secret
```

ZIP launchers include hosted dependencies and add `--hosted` when a nonempty key is
set. Clear the key to return to local mode. An empty key keeps local mode even when
a hosted model name is set. Hosted mode skips the Ollama runtime and model download.

For a source checkout:

```sh
uv sync --locked --dev --extra hosted --python 3.12
uv run uma-bct --hosted --results-directory Results
```

Set the same environment variables first and put test images in `inputs/`.
Direct CLI runs require explicit `--hosted`; environment variables alone do not
select it. `--model` selects local Ollama only. Hosted `setup --hosted` checks local
dependencies without a provider request; it does not validate the key or model's
availability. The first image request checks actual access.

## Interpret results and failures

Stderr reports the selected model and backend. Run details record backend, provider
and model identity; shared detection/transcription stages still use the name `qwen`.
Unavailable token probabilities produce confidence `0` with review flags; they do
not prevent export or prove a transcription is wrong. Review every source and crop.

Requests may retry temporary HTTP 500/502/503/504 failures twice with backoff.
Authentication, model-selection and quota failures are not retried. Errors omit
provider response text and credentials. A Gemini HTTP 400 explicitly rejecting
optional log probabilities receives one retry without them. Keep LiteLLM debug dumps
disabled for private material and keys.

Check the provider console for invalid keys, unavailable models, exhausted quota or
billing limits. A model listing does not establish usable quota. Each photo normally
needs one detection request plus a request per crop; recovery, retries, verification
and AI dates can add requests and cost.

Resume requires the same backend, provider, model and HTTPS origin. Finish an
unfinished batch with its original settings; use `--reprocess` for separate results
after completion. Preserve existing catalogues and inspect the current run's crops.

For containers, see [the optional Docker workflow](../CONTRIBUTING.md#optional-docker-workflow).
