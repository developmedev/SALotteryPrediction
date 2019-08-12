using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using Encog.Engine.Network.Activation;
using Encog.ML.Data.Basic;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;


namespace Prediction
{
    class LottoResult
    {
        public int V1 { get; private set; }
        public int V2 { get; private set; }
        public int V3 { get; private set; }
        public int V4 { get; private set; }
        public int V5 { get; private set; }
        public int V6 { get; private set; }
        public LottoResult(int v1, int v2, int v3, int v4, int v5, int v6)
        {
            V1 = v1;
            V2 = v2;
            V3 = v3;
            V4 = v4;
            V5 = v5;
            V6 = v6;
        }
        public LottoResult(double[] values)
        {
            V1 = (int)Math.Round(values[0]);
            V2 = (int)Math.Round(values[1]);
            V3 = (int)Math.Round(values[2]);
            V4 = (int)Math.Round(values[3]);
            V5 = (int)Math.Round(values[4]);
            V6 = (int)Math.Round(values[5]);
        }
        public bool IsValid()
        {
            return
            V1 >= 1 && V1 <= 49 &&
            V2 >= 1 && V2 <= 49 &&
            V3 >= 1 && V3 <= 49 &&
            V4 >= 1 && V4 <= 49 &&
            V5 >= 1 && V5 <= 49 &&
            V6 >= 1 && V6 <= 49 &&
            V1 != V2 &&
            V1 != V3 &&
            V1 != V4 &&
            V1 != V5 &&
            V1 != V6 &&
            V2 != V3 &&
            V2 != V4 &&
            V2 != V5 &&
            V2 != V6 &&
            V3 != V4 &&
            V3 != V5 &&
            V3 != V6 &&
            V4 != V5 &&
            V4 != V6 &&
            V5 != V6;
        }
        public bool IsOut()
        {
            return
            !(
            V1 >= 1 && V1 <= 49 &&
            V2 >= 1 && V2 <= 49 &&
            V3 >= 1 && V3 <= 49 &&
            V4 >= 1 && V4 <= 49 &&
            V5 >= 1 && V5 <= 49 &&
            V6 >= 1 && V6 <= 49);
        }
        public override string ToString()
        {
            return string.Format(
            "{0},{1},{2},{3},{4},{5}",
            V1, V2, V3, V4, V5, V6);
        }
    }
    class LottoListResults : List<LottoResult> { }
    class Program
    {
        static void Main(string[] args)
        {
            var wc = new WebClient();
            var fileDB = Path.GetTempFileName();
            try
            {
                wc.DownloadFile("http://www.mbnet.com.pl/dl.txt", fileDB);
                LottoListResults dbl = null;
                if (CreateDatabase(fileDB, out dbl))
                {
                    var deep = 20;
                    var network = new BasicNetwork();
                    network.AddLayer(
                    new BasicLayer(null, true, 6 * deep));
                    network.AddLayer(
                    new BasicLayer(
                    new ActivationSigmoid(), true, 5 * 6 * deep));
                    network.AddLayer(
                    new BasicLayer(
                    new ActivationSigmoid(), true, 5 * 6 * deep));
                    network.AddLayer(
                    new BasicLayer(
                    new ActivationLinear(), true, 6));
                    network.Structure.FinalizeStructure();
                    var learningInput = new double[deep][];
                    for (int i = 0; i < deep; ++i)
                    {
                        learningInput[i] = new double[deep * 6];
                        for (int j = 0, k = 0; j < deep; ++j)
                        {
                            var idx = 2 * deep - i - j;
                            var data = dbl[idx];
                            learningInput[i][k++] = (double)data.V1;
                            learningInput[i][k++] = (double)data.V2;
                            learningInput[i][k++] = (double)data.V3;
                            learningInput[i][k++] = (double)data.V4;
                            learningInput[i][k++] = (double)data.V5;
                            learningInput[i][k++] = (double)data.V6;
                        }
                    }
                    var learningOutput = new double[deep][];
                    for (int i = 0; i < deep; ++i)
                    {
                        var idx = deep - 1 - i;
                        var data = dbl[idx];
                        learningOutput[i] = new double[6]
                        {
                            (double)data.V1,
                            (double)data.V2,
                            (double)data.V3,
                            (double)data.V4,
                            (double)data.V5,
                            (double)data.V6
                        };
                    }
                    var trainingSet = new BasicMLDataSet(
                    learningInput,
                    learningOutput
                    );
                    var train = new ResilientPropagation(
                    network, trainingSet);
                    train.NumThreads = Environment.ProcessorCount;
                    START:
                    network.Reset();
                    RETRY:
                    var step = 0;
                    do
                    {
                        train.Iteration();
                        Console.WriteLine("Train Error: {0}", train.Error);
                        ++step;
                    }
                    while (train.Error > 0.001 && step < 20);
                    var passedCount = 0;
                    for (var i = 0; i < deep; ++i)
                    {
                        var should =
                        new LottoResult(learningOutput[i]);
                        var inputn = new BasicMLData(6 * deep);
                        Array.Copy(
                        learningInput[i],
                        inputn.Data,
                        inputn.Data.Length);
                        var comput =
                        new LottoResult(
                        ((BasicMLData)network.
                        Compute(inputn)).Data);
                        var passed = should.ToString() == comput.ToString();
                        if (passed)
                        {
                            Console.ForegroundColor = ConsoleColor.Green;
                            ++passedCount;
                        }
                        else
                        {
                            Console.ForegroundColor = ConsoleColor.Red;
                        }
                        Console.WriteLine("{0} {1} {2} {3}",
                        should.ToString().PadLeft(17, ' '),
                        passed ? "==" : "!=",
                        comput.ToString().PadRight(17, ' '),
                        passed ? "PASS" : "FAIL");
                        Console.ResetColor();
                    }
                    var input = new BasicMLData(6 * deep);
                    for (int i = 0, k = 0; i < deep; ++i)
                    {
                        var idx = deep - 1 - i;
                        var data = dbl[idx];
                        input.Data[k++] = (double)data.V1;
                        input.Data[k++] = (double)data.V2;
                        input.Data[k++] = (double)data.V3;
                        input.Data[k++] = (double)data.V4;
                        input.Data[k++] = (double)data.V5;
                        input.Data[k++] = (double)data.V6;
                    }
                    var perfect = dbl[0];
                    var predict = new LottoResult(
                    ((BasicMLData)network.Compute(input)).Data);
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine("Predict: {0}", predict);
                    Console.ResetColor();
                    if (predict.IsOut())
                        goto START;
                    if ((double)passedCount < (deep * (double)9 / (double)10) ||
                      !predict.IsValid())
                        goto RETRY;
                    Console.WriteLine("Press any key for close...");
                    Console.ReadKey(true);
                }
            }
            catch (Exception exception)
            {
                Console.WriteLine(exception.ToString());
            }
            finally
            {
                File.Delete(fileDB);
            }
        }
        static bool CreateDatabase(
        string fileDB,
        out LottoListResults dbl)
        {
            dbl = new LottoListResults();
            using (var reader = File.OpenText(fileDB))
            {
                var line = string.Empty;
                while ((line = reader.ReadLine()) != null)
                {
                    var values = line.Split(' ')[2].Split(',');
                    var res = new LottoResult(
                    int.Parse(values[0]),
                    int.Parse(values[1]),
                    int.Parse(values[2]),
                    int.Parse(values[3]),
                    int.Parse(values[4]),
                    int.Parse(values[5])
                    );
                    dbl.Add(res);
                }
            }
            dbl.Reverse();
            return true;
        }
    }

}
