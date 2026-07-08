defmodule Lux.LLM.Ollama do
  @moduledoc """
  Ollama LLM implementation that supports passing Beams, Prisms, and Lenses as tools.
  Enables self-hosted LLM capabilities with local model management.
  """

  @behaviour Lux.LLM

  alias Lux.Beam
  alias Lux.Lens
  alias Lux.LLM.ResponseSignal
  alias Lux.Prism

  require Beam
  require Lens
  require Logger

  @doc """
  Default Ollama endpoint. Can be overridden via Config or application env.
  """
  @endpoint "http://localhost:11434/api/chat"

  defmodule Config do
    @moduledoc """
    Configuration module for Ollama.

    ## Fields

    * `:endpoint` - Ollama API endpoint URL (default: `http://localhost:11434/api/chat`)
    * `:model` - Model name to use (default: `llama3.2`)
    * `:api_key` - Optional API key (Ollama doesn't require one by default, but some proxies do)
    * `:temperature` - Sampling temperature (default: `0.7`)
    * `:top_p` - Top-p (nucleus) sampling (default: `0.9`)
    * `:top_k` - Top-k sampling (default: `40`)
    * `:num_ctx` - Context window size (default: `4096`)
    * `:num_predict` - Max tokens to predict (default: `nil`, unlimited)
    * `:repeat_penalty` - Repetition penalty (default: `1.1`)
    * `:stop` - Stop sequences (default: `[]`)
    * `:keep_alive` - How long to keep model loaded in memory (default: `"5m"`)
    * `:format` - Response format - `"json"` for structured output (default: `nil`)
    * `:options` - Raw options passed directly to the Ollama API (default: `%{}`)
    * `:receive_timeout` - Request timeout in ms (default: `300_000` for large model inference)
    * `:system` - System prompt (default: `nil`)
    * `:user` - User identifier (default: `nil`)
    * `:messages` - Prepend messages to the conversation (default: `[]`)
    * `:stream` - Whether to stream responses (default: `false`)
    """
    @type t :: %__MODULE__{
            endpoint: String.t(),
            model: String.t(),
            api_key: String.t() | nil,
            temperature: float(),
            top_p: float(),
            top_k: integer(),
            num_ctx: integer(),
            num_predict: integer() | nil,
            repeat_penalty: float(),
            stop: list(String.t()),
            keep_alive: String.t(),
            format: String.t() | nil,
            options: map(),
            receive_timeout: integer(),
            system: String.t() | nil,
            user: String.t() | nil,
            messages: [map()],
            stream: boolean()
          }

    defstruct endpoint: "http://localhost:11434/api/chat",
              model: "llama3.2",
              api_key: nil,
              temperature: 0.7,
              top_p: 0.9,
              top_k: 40,
              num_ctx: 4096,
              num_predict: nil,
              repeat_penalty: 1.1,
              stop: [],
              keep_alive: "5m",
              format: nil,
              options: %{},
              receive_timeout: 300_000,
              system: nil,
              user: nil,
              messages: [],
              stream: false
  end

  defmodule Models do
    @moduledoc """
    Ollama model management functions.
    Handles listing, pulling, deleting, and checking model status.
    """

    @doc """
    List locally available models.
    Returns `{:ok, [map()]}` with model details or `{:error, reason}`.
    """
    @spec list(String.t(), keyword()) :: {:ok, [map()]} | {:error, term()}
    def list(endpoint \\ default_endpoint(), opts \\ []) do
      req =
        Req.new(
          url: "#{endpoint}/api/tags",
          headers: maybe_auth_headers(opts[:api_key]),
          receive_timeout: Keyword.get(opts, :receive_timeout, 30_000)
        )
        |> Req.merge(Application.get_env(:lux, :ollama_models_req, []))

      case Req.get(req) do
        {:ok, %{status: 200, body: body}} when is_map(body) ->
          {:ok, Map.get(body, "models", [])}

        {:ok, %{status: status, body: body}} ->
          {:error, {:http_error, status, body}}

        {:error, error} ->
          {:error, error}
      end
    end

    @doc """
    Pull (download) a model from the Ollama registry.
    Returns `{:ok, map()}` or `{:error, reason}`.
    """
    @spec pull(String.t(), String.t(), keyword()) :: {:ok, map()} | {:error, term()}
    def pull(model, endpoint \\ default_endpoint(), opts \\ []) do
      body = %{"name" => model, "stream" => false}

      req =
        Req.new(
          url: "#{endpoint}/api/pull",
          json: body,
          headers: maybe_auth_headers(opts[:api_key]),
          receive_timeout: Keyword.get(opts, :receive_timeout, 600_000)
        )
        |> Req.merge(Application.get_env(:lux, :ollama_models_req, []))

      case Req.post(req) do
        {:ok, %{status: 200, body: body}} when is_map(body) ->
          {:ok, body}

        {:ok, %{status: status, body: body}} ->
          {:error, {:http_error, status, body}}

        {:error, error} ->
          {:error, error}
      end
    end

    @doc """
    Delete a local model.
    Returns `:ok` or `{:error, reason}`.
    """
    @spec delete(String.t(), String.t(), keyword()) :: :ok | {:error, term()}
    def delete(model, endpoint \\ default_endpoint(), opts \\ []) do
      body = %{"name" => model}

      req =
        Req.new(
          url: "#{endpoint}/api/delete",
          json: body,
          headers: maybe_auth_headers(opts[:api_key]),
          receive_timeout: Keyword.get(opts, :receive_timeout, 30_000)
        )
        |> Req.merge(Application.get_env(:lux, :ollama_models_req, []))

      case Req.delete(req) do
        {:ok, %{status: 200}} -> :ok
        {:ok, %{status: status, body: body}} -> {:error, {:http_error, status, body}}
        {:error, error} -> {:error, error}
      end
    end

    @doc """
    Show information about a model including size, parameters, and family.
    Returns `{:ok, map()}` or `{:error, reason}`.
    """
    @spec show(String.t(), String.t(), keyword()) :: {:ok, map()} | {:error, term()}
    def show(model, endpoint \\ default_endpoint(), opts \\ []) do
      req =
        Req.new(
          url: "#{endpoint}/api/show",
          json: %{"name" => model},
          headers: maybe_auth_headers(opts[:api_key]),
          receive_timeout: Keyword.get(opts, :receive_timeout, 30_000)
        )
        |> Req.merge(Application.get_env(:lux, :ollama_models_req, []))

      case Req.post(req) do
        {:ok, %{status: 200, body: body}} when is_map(body) -> {:ok, body}
        {:ok, %{status: status, body: body}} -> {:error, {:http_error, status, body}}
        {:error, error} -> {:error, error}
      end
    end

    @doc """
    Check if the Ollama server is running and accessible.
    Returns `:ok` or `{:error, reason}`.
    """
    @spec running?(String.t(), keyword()) :: :ok | {:error, term()}
    def running?(endpoint \\ default_endpoint(), opts \\ []) do
      req =
        Req.new(
          url: endpoint,
          headers: maybe_auth_headers(opts[:api_key]),
          receive_timeout: Keyword.get(opts, :receive_timeout, 5_000)
        )
        |> Req.merge(Application.get_env(:lux, :ollama_models_req, []))

      case Req.get(req) do
        {:ok, %{status: 200}} -> :ok
        {:ok, %{status: status}} -> {:error, {:not_running, status}}
        {:error, error} -> {:error, {:not_running, error}}
      end
    end

    defp default_endpoint do
      Application.get_env(:lux, :ollama_endpoint, "http://localhost:11434")
    end

    defp maybe_auth_headers(nil), do: []
    defp maybe_auth_headers(key), do: [{"Authorization", "Bearer #{key}"}]
  end

  @impl true
  def call(prompt, tools, config) do
    config =
      struct(
        Config,
        Map.merge(
          %{
            model: Application.get_env(:lux, :ollama_models)[:default],
            api_key: Application.get_env(:lux, :api_keys)[:ollama]
          },
          config
        )
      )

    messages = config.messages ++ build_messages(config.system, prompt)
    tools_config = build_tools_config(tools)

    body =
      %{
        model: Lux.Config.resolve(config.model),
        messages: messages,
        stream: false
      }
      |> maybe_add_options(config)
      |> maybe_add_tools(tools_config)
      |> maybe_add_format(config)
      |> maybe_add_keep_alive(config)

    req_opts =
      [
        url: config.endpoint,
        json: body,
        headers: build_headers(config.api_key),
        receive_timeout: config.receive_timeout
      ]
      |> Keyword.merge(Application.get_env(:lux, __MODULE__, []))

    req_opts
    |> Req.new()
    |> Req.post()
    |> case do
      {:ok, %{status: 200} = response} ->
        handle_response(response, config)

      {:ok, %{status: 404}} ->
        {:error, :model_not_found}

      {:ok, %{status: 503}} ->
        {:error, :ollama_not_running}

      {:ok, %{status: status, body: %{"error" => message}}} ->
        {:error, {status, message}}

      {:ok, %{status: status, body: body}} ->
        {:error, {:unexpected_response, status, body}}

      {:error, error} ->
        handle_error(error)
    end
  end

  defp build_headers(nil), do: [{"Content-Type", "application/json"}]

  defp build_headers(api_key),
    do: [{"Authorization", "Bearer #{api_key}"}, {"Content-Type", "application/json"}]

  defp build_messages(nil, prompt), do: [%{role: "user", content: prompt}]

  defp build_messages(system, prompt) do
    [%{role: "system", content: system}, %{role: "user", content: prompt}]
  end

  defp build_tools_config([]), do: []
  defp build_tools_config(tools), do: Enum.map(tools, &tool_to_function/1)

  defp maybe_add_options(body, config) do
    options =
      config.options
      |> Map.merge(%{
        temperature: config.temperature,
        top_p: config.top_p,
        top_k: config.top_k,
        num_ctx: config.num_ctx,
        repeat_penalty: config.repeat_penalty
      })
      |> maybe_add_num_predict(config.num_predict)
      |> maybe_add_stop(config.stop)
      |> Enum.reject(fn {_k, v} -> is_nil(v) end)
      |> Map.new()

    Map.put(body, :options, options)
  end

  defp maybe_add_num_predict(options, nil), do: options
  defp maybe_add_num_predict(options, n), do: Map.put(options, :num_predict, n)

  defp maybe_add_stop(options, []), do: options
  defp maybe_add_stop(options, stop), do: Map.put(options, :stop, stop)

  defp maybe_add_tools(body, []), do: body

  defp maybe_add_tools(body, tools) do
    Map.put(body, :tools, tools)
  end

  defp maybe_add_format(body, %Config{format: "json"}) do
    Map.put(body, :format, "json")
  end

  defp maybe_add_format(body, _), do: body

  defp maybe_add_keep_alive(body, %Config{keep_alive: keep_alive})
       when is_binary(keep_alive) and keep_alive != "" do
    Map.put(body, :keep_alive, keep_alive)
  end

  defp maybe_add_keep_alive(body, _), do: body

  # Tool conversion - same pattern as OpenAI/TogetherAI

  def tool_to_function({:python, path}) do
    path
    |> Prism.view()
    |> tool_to_function()
  end

  def tool_to_function(tool_module) when is_atom(tool_module) and not is_nil(tool_module) do
    cond do
      Lux.prism?(tool_module) ->
        tool_to_function(tool_module.view())

      Lux.beam?(tool_module) ->
        tool_to_function(tool_module.view())

      Lux.lens?(tool_module) ->
        tool_to_function(tool_module.view())

      true ->
        raise "Unsupported tool type: #{inspect(tool_module)}"
    end
  end

  def tool_to_function(%Beam{name: name, description: description, input_schema: input_schema}) do
    %{
      type: "function",
      function: %{
        name: String.replace(name || "unnamed_beam", ".", "_"),
        description: description || "",
        parameters: input_schema
      }
    }
  end

  def tool_to_function(%Prism{module_name: name, description: description, input_schema: input_schema}) do
    %{
      type: "function",
      function: %{
        name: String.replace(name, ".", "_"),
        description: description || "",
        parameters: input_schema
      }
    }
  end

  def tool_to_function(%Lens{module_name: name, description: description, schema: schema}) do
    %{
      type: "function",
      function: %{
        name: String.replace(name || "unnamed_lens", ".", "_"),
        description: description || "",
        parameters: schema
      }
    }
  end

  # Response handling

  defp handle_response(%{body: body}, _config) when is_map(body) do
    # Ollama non-streaming response has "message" at top level
    case body do
      %{"message" => message} ->
        with {:ok, content} <- parse_content(message["content"]),
             {:ok, tool_calls_results} <- execute_tool_calls(message["tool_calls"]) do
          payload = %{
            content: content,
            model: body["model"],
            finish_reason: if(message["done"] == true, do: "stop", else: nil),
            tool_calls: message["tool_calls"],
            tool_calls_results: tool_calls_results
          }

          metadata = %{
            id: nil,
            created: nil,
            usage: %{
              prompt_tokens: body["prompt_eval_count"],
              completion_tokens: body["eval_count"],
              total_tokens: nil
            },
            system_fingerprint: nil
          }

          %{
            schema_id: ResponseSignal,
            payload: payload,
            metadata: metadata
          }
          |> Lux.Signal.new()
          |> ResponseSignal.validate()
        end

      # Standard OpenAI-compatible format (some Ollama setups proxy this way)
      %{"choices" => [choice | _]} ->
        %{"message" => message} = choice

        with {:ok, content} <- parse_content(message["content"]),
             {:ok, tool_calls_results} <- execute_tool_calls(message["tool_calls"]) do
          payload = %{
            content: content,
            model: body["model"],
            finish_reason: choice["finish_reason"],
            tool_calls: message["tool_calls"],
            tool_calls_results: tool_calls_results
          }

          metadata = %{
            id: body["id"],
            created: body["created"],
            usage: body["usage"],
            system_fingerprint: body["system_fingerprint"]
          }

          %{
            schema_id: ResponseSignal,
            payload: payload,
            metadata: metadata
          }
          |> Lux.Signal.new()
          |> ResponseSignal.validate()
        end

      _ ->
        {:error, {:unexpected_response_format, body}}
    end
  end

  def parse_content(content) when is_binary(content) do
    case Jason.decode(content) do
      {:ok, structured_output} ->
        {:ok, structured_output}

      {:error, _} ->
        {:error, "failed to parse content: #{inspect(content)}"}
    end
  end

  def parse_content(_), do: {:ok, nil}

  def execute_tool_calls(tool_calls) when is_list(tool_calls) do
    tool_calls
    |> Enum.map(&execute_tool_call/1)
    |> Enum.reduce({:ok, []}, fn
      {:ok, result, _log}, {:ok, results} ->
        {:ok, [result | results]}

      {:ok, result}, {:ok, results} ->
        {:ok, [result | results]}

      error, _ ->
        error
    end)
  end

  def execute_tool_calls(nil), do: {:ok, nil}

  def execute_tool_call(%{"function" => %{"name" => tool_name, "arguments" => args}}) do
    args = Jason.decode!(args)
    execute_tool(tool_name, args, nil)
  end

  def execute_tool(tool_name, args, ctx) when is_binary(tool_name) do
    tool_name
    |> String.replace("_", ".")
    |> List.wrap()
    |> Module.concat()
    |> Code.ensure_loaded()
    |> case do
      {:module, module_name} ->
        execute_tool(module_name, args, ctx)

      {:error, :nofile} ->
        {:error,
         "Failed to load tool module #{tool_name}: It doesn't seems to be implemented or reacheable"}

      {:error, error} ->
        {:error, "Failed to load tool module #{tool_name}: #{inspect(error)}"}
    end
  end

  def execute_tool(tool_module, args, ctx) when is_atom(tool_module) do
    cond do
      Lux.prism?(tool_module) ->
        tool_module.handler(args, ctx)

      Lux.beam?(tool_module) ->
        tool_module.run(args, ctx)

      Lux.lens?(tool_module) ->
        tool_module.focus(args)

      true ->
        {:error,
         """
         Tool #{tool_module} does not seem to be a valid Beam or Prism
         as it does not have a registered `handler` or `run` function.
         """}
    end
  end

  defp handle_error(error) do
    Logger.error("Ollama API error: #{inspect(error)}")
    {:error, "Ollama API error: #{inspect(error)}"}
  end
end
