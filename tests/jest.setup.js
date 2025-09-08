jest.mock('mongoose', () => {
  const actualMongoose = jest.requireActual('mongoose');
  const fakeConnection = {
    on: jest.fn(),
    once: jest.fn(),
    close: jest.fn(),
    readyState: 1,
  };
  return {
    ...actualMongoose,
    connect: jest.fn().mockResolvedValue({}),
    createConnection: jest.fn().mockReturnValue(fakeConnection),
    connection: fakeConnection,
    Schema: actualMongoose.Schema,
    model: actualMongoose.model,
    Types: actualMongoose.Types,
  };
});
